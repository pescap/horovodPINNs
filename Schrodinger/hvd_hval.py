"""
@author: Maziar Raissi
"""

import sys

sys.path.insert(0, "../PINNs/Utilities/")

import horovod.tensorflow as hvd

hvd.init()

import tensorflow.compat.v1 as tf

tf.compat.v1.disable_eager_execution()
tf.disable_v2_behavior()

import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
from plotting import newfig, savefig
from mpl_toolkits.mplot3d import Axes3D
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
import datetime

now = datetime.datetime.now()
parser = argparse.ArgumentParser(description="Set parameters")

parser.add_argument("--N", default=1000, type=int)
parser.add_argument("--epochs", default=30000, type=int)
parser.add_argument("--seed", default=1234, type=int)

parser.add_argument(
    "--fp16-allreduce",
    action="store_true",
    default=False,
    help="use fp16 compression during allreduce",
)

parser.add_argument(
    "--num-warmup-batches",
    type=int,
    default=10,
    help="number of warm-up batches that don't count towards benchmark",
)

parser.add_argument(
    "--use-adasum",
    action="store_true",
    default=False,
    help="use adasum algorithm to do reduction",
)

parser.add_argument(
    "--use-scaler", action="store_true", default=False, help="use scaler"
)

parser.add_argument("--backward", type=int, default=1, help="backward passes per step")

parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")

parser.add_argument(
    "--save",
    action="store_true",
    default=False,
    help="save results",
)

args = parser.parse_args()
epochs = args.epochs
seed = args.seed
N = args.N

seed_rank = seed + 1000 * hvd.local_rank()

np.random.seed(seed_rank)
tf.set_random_seed(seed)

size = hvd.size()
rank = hvd.local_rank()


config = tf.ConfigProto()
config.gpu_options.visible_device_list = str(hvd.local_rank())
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = True

sess = tf.Session(config=config)

name = (
    "results/"
    + "seed"
    + str(seed)
    + "_N"
    + str(N)
    + "_epochs"
    + str(epochs)
    + "_size"
    + str(hvd.size())
)
if args.use_adasum:
    name = name + "_adasum"

if args.fp16_allreduce:
    name = name + "_fp16"


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x0, u0, v0, tb, X_f, layers, lb, ub, X_f_test):

        X0 = np.concatenate((x0, 0 * x0), 1)  # (x0, 0)
        X_lb = np.concatenate((0 * tb + lb[0], tb), 1)  # (lb[0], tb)
        X_ub = np.concatenate((0 * tb + ub[0], tb), 1)  # (ub[0], tb)

        self.lb = lb
        self.ub = ub

        self.x0 = X0[:, 0:1]
        self.t0 = X0[:, 1:2]

        self.x_lb = X_lb[:, 0:1]
        self.t_lb = X_lb[:, 1:2]

        self.x_ub = X_ub[:, 0:1]
        self.t_ub = X_ub[:, 1:2]

        self.x_f = X_f[:, 0:1]
        self.t_f = X_f[:, 1:2]

        self.x_f_test = X_f_test[:, 0:1]
        self.t_f_test = X_f_test[:, 1:2]

        self.X_f_test = X_f_test

        self.u0 = u0
        self.v0 = v0

        # Initialize NNs
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        # tf Placeholders
        self.x0_tf = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])
        self.t0_tf = tf.placeholder(tf.float32, shape=[None, self.t0.shape[1]])

        self.u0_tf = tf.placeholder(tf.float32, shape=[None, self.u0.shape[1]])
        self.v0_tf = tf.placeholder(tf.float32, shape=[None, self.v0.shape[1]])

        self.x_lb_tf = tf.placeholder(tf.float32, shape=[None, self.x_lb.shape[1]])
        self.t_lb_tf = tf.placeholder(tf.float32, shape=[None, self.t_lb.shape[1]])

        self.x_ub_tf = tf.placeholder(tf.float32, shape=[None, self.x_ub.shape[1]])
        self.t_ub_tf = tf.placeholder(tf.float32, shape=[None, self.t_ub.shape[1]])

        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

        # tf Graphs
        self.u0_pred, self.v0_pred, _, _ = self.net_uv(self.x0_tf, self.t0_tf)
        (
            self.u_lb_pred,
            self.v_lb_pred,
            self.u_x_lb_pred,
            self.v_x_lb_pred,
        ) = self.net_uv(self.x_lb_tf, self.t_lb_tf)
        (
            self.u_ub_pred,
            self.v_ub_pred,
            self.u_x_ub_pred,
            self.v_x_ub_pred,
        ) = self.net_uv(self.x_ub_tf, self.t_ub_tf)
        self.f_u_pred, self.f_v_pred = self.net_f_uv(self.x_f_tf, self.t_f_tf)

        # Loss
        self.loss_0 = tf.reduce_mean(
            tf.square(self.u0_tf - self.u0_pred)
        ) + tf.reduce_mean(tf.square(self.v0_tf - self.v0_pred))

        self.loss_b = (
            tf.reduce_mean(tf.square(self.u_lb_pred - self.u_ub_pred))
            + tf.reduce_mean(tf.square(self.v_lb_pred - self.v_ub_pred))
            + tf.reduce_mean(tf.square(self.u_x_lb_pred - self.u_x_ub_pred))
            + tf.reduce_mean(tf.square(self.v_x_lb_pred - self.v_x_ub_pred))
        )

        self.loss_f = tf.reduce_mean(tf.square(self.f_u_pred)) + tf.reduce_mean(
            tf.square(self.f_v_pred)
        )

        lr_scaler = hvd.size()
        # By default, Adasum doesn't need scaling when increasing batch size. If used with NCCL,
        # scale lr by local_size

        if args.use_adasum and args.use_scaler:
            lr_scaler = hvd.local_size()
        elif args.use_scaler:
            lr_scaler = hvd.size()
        else:
            lr_scaler = 1

        if hvd.local_rank() == 0:
            print(lr_scaler, "LR_SCALER")

        optimizer = tf.train.AdamOptimizer(args.lr * lr_scaler)  # * size)
        compression = (
            hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
        )

        optimizer = hvd.DistributedOptimizer(
            optimizer,
            compression=compression,
            op=hvd.Adasum if args.use_adasum else hvd.Average,
            backward_passes_per_step=args.backward,
        )

        loss_weights = [1, 1, 1]
        self.train_op_Adam = optimizer.minimize(
            self.loss_f * loss_weights[0]
            + self.loss_b * loss_weights[1]
            + self.loss_0 * loss_weights[2]
        )

        init = tf.global_variables_initializer()
        sess.run(init)

        bcast = hvd.broadcast_global_variables(0)
        sess.run(bcast)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(
                tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32
            )
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(
            tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev),
            dtype=tf.float32,
        )

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_uv(self, x, t):
        X = tf.concat([x, t], 1)

        uv = self.neural_net(X, self.weights, self.biases)
        u = uv[:, 0:1]
        v = uv[:, 1:2]

        u_x = tf.gradients(u, x)[0]
        v_x = tf.gradients(v, x)[0]

        return u, v, u_x, v_x

    def net_f_uv(self, x, t):
        u, v, u_x, v_x = self.net_uv(x, t)

        u_t = tf.gradients(u, t)[0]
        u_xx = tf.gradients(u_x, x)[0]

        v_t = tf.gradients(v, t)[0]
        v_xx = tf.gradients(v_x, x)[0]

        f_u = u_t + 0.5 * v_xx + (u**2 + v**2) * v
        f_v = v_t - 0.5 * u_xx - (u**2 + v**2) * u

        return f_u, f_v

    def callback(self, loss):
        print("Loss:", loss)

    def train(self, nIter):

        train_list = []
        test_list = []
        metrics_list = []
        time_list = []
        pointsec_list = []
        tf_dict = {
            self.x0_tf: self.x0,
            self.t0_tf: self.t0,
            self.u0_tf: self.u0,
            self.v0_tf: self.v0,
            self.x_lb_tf: self.x_lb,
            self.t_lb_tf: self.t_lb,
            self.x_ub_tf: self.x_ub,
            self.t_ub_tf: self.t_ub,
            self.x_f_tf: self.x_f,
            self.t_f_tf: self.t_f,
        }

        tf_dict_test = {
            self.x0_tf: self.x0,
            self.t0_tf: self.t0,
            self.u0_tf: self.u0,
            self.v0_tf: self.v0,
            self.x_lb_tf: self.x_lb,
            self.t_lb_tf: self.t_lb,
            self.x_ub_tf: self.x_ub,
            self.t_ub_tf: self.t_ub,
            self.x_f_tf: self.x_f_test,
            self.t_f_tf: self.t_f_test,
        }

        ta = time.perf_counter()
        for it in range(nIter):
            sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 500 == 0 and hvd.rank() == 0:
                texec = time.perf_counter() - ta
                loss_f = sess.run(self.loss_f, tf_dict)
                loss_b = sess.run(self.loss_b, tf_dict)
                loss_0 = sess.run(self.loss_0, tf_dict)

                loss_f_test = sess.run(self.loss_f, tf_dict_test)
                loss_b_test = sess.run(self.loss_b, tf_dict_test)
                loss_0_test = sess.run(self.loss_0, tf_dict_test)

                loss_value = loss_f + loss_b + loss_0
                loss_value_test = loss_f_test + loss_b_test + loss_0_test

                u_pred, v_pred, f_u_pred, f_v_pred = self.predict(self.X_f_test)
                h_pred = np.sqrt(u_pred**2 + v_pred**2)

                error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
                error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
                error_h = np.linalg.norm(h_star - h_pred, 2) / np.linalg.norm(h_star, 2)

                error_l2 = [error_u, error_v, error_h]
                result = [f"{item:.3f}" for item in error_l2]

                pointsec = N * 500 / texec

                print(
                    "It: %d, Loss: %.1e, Loss_test: %.1e, Time: %.2f, p/s :%.1e"
                    % (it, loss_value, loss_value_test, texec, pointsec),
                    "Error l2: ",
                    result,
                )

                train_list.append(loss_value)
                test_list.append(loss_value_test)
                metrics_list.append(error_l2)
                time_list.append(texec)
                pointsec_list.append(pointsec)

                ta = time.perf_counter()

        self.train_list = train_list
        self.test_list = test_list
        self.metrics_list = metrics_list
        self.time_list = time_list
        self.pointsec_list = pointsec_list

    def predict(self, X_star):

        tf_dict = {self.x0_tf: X_star[:, 0:1], self.t0_tf: X_star[:, 1:2]}

        u_star = sess.run(self.u0_pred, tf_dict)
        v_star = sess.run(self.v0_pred, tf_dict)

        tf_dict = {self.x_f_tf: X_star[:, 0:1], self.t_f_tf: X_star[:, 1:2]}

        f_u_star = sess.run(self.f_u_pred, tf_dict)
        f_v_star = sess.run(self.f_v_pred, tf_dict)

        return u_star, v_star, f_u_star, f_v_star


noise = 0.0

# Domain bounds
lb = np.array([-5.0, 0.0])
## TEST WITH RANK

ub = np.array([5.0, np.pi / 2])

N0 = 200
N_b = 200
N_f = N

layers = [2, 100, 100, 100, 100, 2]

data = scipy.io.loadmat("../Data/NLS.mat")

t = data["tt"].flatten()[:, None]
x = data["x"].flatten()[:, None]

Exact = data["uu"]
Exact_u = np.real(Exact)
Exact_v = np.imag(Exact)
Exact_h = np.sqrt(Exact_u**2 + Exact_v**2)


X, T = np.meshgrid(x, t)

X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_star = Exact_u.T.flatten()[:, None]
v_star = Exact_v.T.flatten()[:, None]
h_star = Exact_h.T.flatten()[:, None]

idx_x = np.random.choice(x.shape[0], N0, replace=False)
x0 = x[idx_x, :]
u0 = Exact_u[idx_x, 0:1]
v0 = Exact_v[idx_x, 0:1]

idx_t = np.random.choice(t.shape[0], N_b, replace=False)
tb = t[idx_t, :]

X_f = lb + (ub - lb) * lhs(2, N_f)

model = PhysicsInformedNN(x0, u0, v0, tb, X_f, layers, lb, ub, X_star)


start_time = time.time()
model.train(epochs)

if hvd.rank() == 0:
    elapsed = time.time() - start_time
    theta = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

    print("Training time: %.4f" % (elapsed))
    print(theta, "# Parameters:")

    u_pred, v_pred, f_u_pred, f_v_pred = model.predict(X_star)
    h_pred = np.sqrt(u_pred**2 + v_pred**2)

    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
    error_h = np.linalg.norm(h_star - h_pred, 2) / np.linalg.norm(h_star, 2)
    print("Error u: %e" % (error_u))
    print("Error v: %e" % (error_v))
    print("Error h: %e" % (error_h))

    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method="cubic")
    V_pred = griddata(X_star, v_pred.flatten(), (X, T), method="cubic")
    H_pred = griddata(X_star, h_pred.flatten(), (X, T), method="cubic")

    FU_pred = griddata(X_star, f_u_pred.flatten(), (X, T), method="cubic")
    FV_pred = griddata(X_star, f_v_pred.flatten(), (X, T), method="cubic")

    print(N_f, "N_f")

    train_list = np.array(model.train_list)
    test_list = np.array(model.test_list)
    metrics_list = np.array(model.metrics_list)
    time_list = np.array(model.time_list)
    pointsec_list = np.array(model.pointsec_list)

    pointsec_mean = np.mean(pointsec_list[3:])
    pointsec_conf = 1.96 * np.std(pointsec_list[3:])

    time_mean = np.mean(time_list[3:])
    time_conf = 1.96 * np.std(time_list[3:])

    print("Time for 500 epochs:= ", time_mean, "+-", time_conf)
    print("Pointsec for 500 epochs:= ", pointsec_mean, "+-", pointsec_conf)

    niter = time_list.shape[0]
    my_dict = {
        "texec": elapsed,
        "theta": theta,
        "err": [error_u, error_v, error_h],
        "train": train_list,
        "test": test_list,
        "metrics": metrics_list,
        "times": time_list,
        "pointsec": pointsec_list,
        "date": now,
    }
    if args.save:
        np.save(name + ".npy", my_dict)
