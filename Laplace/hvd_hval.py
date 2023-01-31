import os
import tensorflow.compat.v1 as tf

tf.compat.v1.disable_eager_execution()
tf.disable_v2_behavior()

import horovod.tensorflow as hvd

import numpy as np
import time
import matplotlib.pyplot as plt
import os
import argparse
import time

now = datetime.datetime.now()
parser = argparse.ArgumentParser(description="Set parameters")

parser.add_argument("--N", default=16, type=int)
parser.add_argument("--epochs", default=10000, type=int)
parser.add_argument("--seed", default=1234, type=int)
parser.add_argument(
    "--fp16-allreduce",
    action="store_true",
    default=False,
    help="use fp16 compression during allreduce",
)

parser.add_argument(
    "--use-adasum",
    action="store_true",
    default=False,
    help="use adasum algorithm to do reduction",
)

parser.add_argument(
    "--save",
    action="store_true",
    default=False,
    help="save results",
)


parser.add_argument("--backward", type=int, default=1, help="backward passes per step")


parser.add_argument("--lr", type=float, default=0.001, help="learning rate")


args = parser.parse_args()
epochs = args.epochs
seed = args.seed
N = args.N

lr = args.lr

hvd.init()

#os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())

seed_rank = seed + 1000 * hvd.local_rank()

np.random.seed(seed_rank)
tf.set_random_seed(seed)


config = tf.ConfigProto()
config.gpu_options.visible_device_list = str(hvd.local_rank())
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


name = "results/" + "seed" + str(seed) + "_N" + str(N) + "_epochs" + str(epochs) + "_size" + str(hvd.size())
if args.use_adasum:
    name = name + '_adasum'

if args.fp16_allreduce:
    name = name + '_fp16'
    

def hyper_initial(size):
    in_dim = size[0]
    out_dim = size[1]
    std = np.sqrt(2.0 / (in_dim + out_dim))
    return tf.Variable(tf.random_normal(shape=size, stddev=std))


def DNN(X, W, b):
    A = X
    L = len(W)
    for i in range(L - 1):
        A = tf.tanh(tf.add(tf.matmul(A, W[i]), b[i]))
    Y = tf.add(tf.matmul(A, W[-1]), b[-1])
    return Y


def pdenn(X, W, b):
    u = DNN(X, W, b)
    u_x = tf.gradients(u, X)[0]
    u_xx = tf.gradients(u_x, X)[0]
    f = tf.sin(np.pi * X) * np.pi * np.pi
    R = u_xx + f

    return R


x_col = (np.random.random(N) * 8 - 1).reshape((-1, 1))

print(
    "This is from rank %d with x_col min and max as %.2e and %.3e"
    % (hvd.rank(), x_col.min(), x_col.max())
)

x_0 = np.array([-1]).reshape((-1, 1))
x_1 = np.array([7]).reshape((-1, 1))
y_0 = np.sin(np.pi * x_0)
y_1 = np.sin(np.pi * x_1)

layers = [1] + 4 * [50] + [1]
L = len(layers)
W = [hyper_initial([layers[l - 1], layers[l]]) for l in range(1, L)]
b = [tf.Variable(tf.zeros([1, layers[l]])) for l in range(1, L)]

x_train = tf.placeholder(tf.float32, shape=[None, 1])
x_0_train = tf.placeholder(tf.float32, shape=[None, 1])
y_0_train = tf.placeholder(tf.float32, shape=[None, 1])
x_1_train = tf.placeholder(tf.float32, shape=[None, 1])
y_1_train = tf.placeholder(tf.float32, shape=[None, 1])
y_nn = DNN(x_train, W, b)
y_0_nn = DNN(x_0_train, W, b)
y_1_nn = DNN(x_1_train, W, b)
R_nn = pdenn(x_train, W, b)

loss = (
    tf.reduce_mean(tf.square(R_nn))
    + tf.reduce_mean(tf.square(y_0_nn - y_0_train))
    + tf.reduce_mean(tf.square(y_1_nn - y_1_train))
)

optimizer = tf.train.AdamOptimizer(lr)


# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

optimizer = hvd.DistributedOptimizer(
    optimizer,
    compression=compression,
    op=hvd.Adasum if args.use_adasum else hvd.Average,
    backward_passes_per_step=args.backward,
)

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

bcast = hvd.broadcast_global_variables(0)
sess.run(bcast)

x_test = (np.random.random(N) * 8 - 1).reshape((-1, 1))
y_exact = np.sin(np.pi * x_test)

train_dict = {
    x_train: x_col,
    x_0_train: x_0,
    y_0_train: y_0,
    x_1_train: x_1,
    y_1_train: y_1,
}
test_dict = {
    x_train: x_test,
    x_0_train: x_0,
    y_0_train: y_0,
    x_1_train: x_1,
    y_1_train: y_1,
}

start_time = time.perf_counter()
n = 0

train_list = []
test_list = []
metrics_list = []
time_list = []
pointsec_list = []

ta = time.perf_counter()
while n <= epochs:
    y_pred, train_, loss_ = sess.run([y_nn, train, loss], feed_dict=train_dict)
    y_test, test_, loss_test = sess.run([y_nn, train, loss], feed_dict=test_dict)
    err_l2 = np.linalg.norm((y_test - y_exact) / np.linalg.norm(y_exact))

    if n % 500 == 0 and hvd.rank() == 0:
        texec = time.perf_counter() - ta
        pointsec = N * 500 / texec
        print(
            "n = %d, loss = %.3e, loss test = %.3e, metrics = %.3e, time = %.1e, p/s = %.3e"
            % (n, loss_, loss_test, err_l2, texec, pointsec)
        )
        train_list.append(loss_)
        test_list.append(loss_test)
        metrics_list.append(err_l2)
        time_list.append(texec)
        pointsec_list.append(pointsec)
        ta = time.perf_counter()

    n += 1

if hvd.rank() == 0:

    theta = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    stop_time = time.perf_counter()
    texec = stop_time - start_time

    train_list = np.array(train_list)
    test_list = np.array(test_list)
    metrics_list = np.array(metrics_list)
    time_list = np.array(time_list)
    pointsec_list = np.array(pointsec_list)

    pointsec_mean = np.mean(pointsec_list[3:])
    pointsec_conf = 1.96 * np.std(pointsec_list[3:])
    
    time_mean = np.mean(time_list[3:])
    time_conf = 1.96 * np.std(time_list[3:])
    
    min_index = np.array(test_list).argmin()
    err_l2 = np.array(metrics_list)[min_index]

    print("psec per GPU: %.1f +-%.1f" % (pointsec_mean, pointsec_conf))
    print(err_l2, "L2-norm")
    print(texec, "texec")
    print(theta, "# Parameters:")
    
    print('Time for 500 epochs:= ', time_mean, '+-', time_conf)
    print('Pointsec for 500 epochs:= ', pointsec_mean, '+-', pointsec_conf)

    niter = time_list.shape[0]
    
    my_dict = {       
        "texec": texec,
        "theta": theta,
        "err": err_l2,
        "train": train_list,
        "test": test_list,
        "metrics": metrics_list,
        "times": time_list,
        "pointsec": pointsec_list,
        "date": now
    }
    if args.save:
        np.save(name + ".npy", my_dict)
