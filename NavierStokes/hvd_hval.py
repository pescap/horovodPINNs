import sys

sys.path.insert(0, "../PINNs/Utilities/")

import tensorflow.compat.v1 as tf

tf.compat.v1.disable_eager_execution()
tf.disable_v2_behavior()

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import argparse

import horovod.tensorflow as hvd
import datetime

now = datetime.datetime.now()

parser = argparse.ArgumentParser(description="Set parameters")
parser.add_argument("--N", default=5000, type=int)
parser.add_argument("--epochs", default=100000, type=int)
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

parser.add_argument("--backward", type=int, default=1, help="backward passes per step")

parser.add_argument("--lr", type=float, default=0.001, help="learning rate")

parser.add_argument(
    "--save",
    action="store_true",
    default=False,
    help="save results",
)

args = parser.parse_args()
epochs = args.epochs
seed = args.seed
N_train = args.N

hvd.init()

name = (
    "results/"
    + "seed"
    + str(seed)
    + "_N"
    + str(N_train)
    + "_epochs"
    + str(epochs)
    + "_size"
    + str(hvd.size())
)
if args.use_adasum:
    name = name + "_adasum"

if args.fp16_allreduce:
    name = name + "_fp16"


seed_rank = seed + 1000 * hvd.local_rank()

np.random.seed(seed_rank)
tf.set_random_seed(seed)


class PhysicsInformedNN:
    # Initialize the class
    def __init__(
        self, x, y, t, u, v, x_test, y_test, t_test, u_test, v_test, p_test, layers
    ):

        X = np.concatenate([x, y, t], 1)
        X_test = np.concatenate([x_test, y_test, t_test], 1)

        self.lb = X.min(0)
        self.ub = X.max(0)

        self.X = X

        self.x = X[:, 0:1]
        self.y = X[:, 1:2]
        self.t = X[:, 2:3]

        self.x_test = X_test[:, 0:1]
        self.y_test = X_test[:, 1:2]
        self.t_test = X_test[:, 2:3]

        self.u = u
        self.v = v

        self.u_test = u_test
        self.v_test = v_test
        self.p_test = p_test

        self.layers = layers

        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)

        # Initialize parameters
        self.lambda_1 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_2 = tf.Variable([0.0], dtype=tf.float32)

        # tf placeholders and graph
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = str(hvd.local_rank())
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True

        self.sess = tf.Session(config=config)

        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])

        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.v_tf = tf.placeholder(tf.float32, shape=[None, self.v.shape[1]])

        (
            self.u_pred,
            self.v_pred,
            self.p_pred,
            self.f_u_pred,
            self.f_v_pred,
        ) = self.net_NS(self.x_tf, self.y_tf, self.t_tf)

        self.loss = (
            tf.reduce_sum(tf.square(self.u_tf - self.u_pred))
            + tf.reduce_sum(tf.square(self.v_tf - self.v_pred))
            + tf.reduce_sum(tf.square(self.f_u_pred))
            + tf.reduce_sum(tf.square(self.f_v_pred))
        )

        # self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
        #                                                        method = 'L-BFGS-B',
        #                                                        options = {'maxiter': 50000,
        #                                                                   'maxfun': 50000,
        #                                                                   'maxcor': 50,
        #                                                                   'maxls': 50,
        #                                                                   'ftol' : 1.0 * np.finfo(float).eps})

        # self.optimizer_Adam = tf.train.AdamOptimizer(args.lr)
        optimizer = tf.train.AdamOptimizer(args.lr)
        compression = (
            hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
        )
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            compression=compression,
            op=hvd.Adasum if args.use_adasum else hvd.Average,
            backward_passes_per_step=args.backward,
        )

        self.optimizer_Adam = optimizer
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        bcast = hvd.broadcast_global_variables(0)
        self.sess.run(bcast)

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

    def net_NS(self, x, y, t):
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2

        psi_and_p = self.neural_net(tf.concat([x, y, t], 1), self.weights, self.biases)
        psi = psi_and_p[:, 0:1]
        p = psi_and_p[:, 1:2]

        u = tf.gradients(psi, y)[0]
        v = -tf.gradients(psi, x)[0]

        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]

        v_t = tf.gradients(v, t)[0]
        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        v_xx = tf.gradients(v_x, x)[0]
        v_yy = tf.gradients(v_y, y)[0]

        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]

        f_u = u_t + lambda_1 * (u * u_x + v * u_y) + p_x - lambda_2 * (u_xx + u_yy)
        f_v = v_t + lambda_1 * (u * v_x + v * v_y) + p_y - lambda_2 * (v_xx + v_yy)

        return u, v, p, f_u, f_v

    def callback(self, loss, lambda_1, lambda_2):
        print("Loss: %.3e, l1: %.3f, l2: %.5f" % (loss, lambda_1, lambda_2))

    def train(self, nIter):
        train_list = []
        test_list = []
        metrics_u_list = []
        metrics_v_list = []
        metrics_p_list = []
        time_list = []
        pointsec_list = []
        lambda_1_list = []
        lambda_2_list = []

        tf_dict = {
            self.x_tf: self.x,
            self.y_tf: self.y,
            self.t_tf: self.t,
            self.u_tf: self.u,
            self.v_tf: self.v,
        }

        tf_test_dict = {
            self.x_tf: self.x_test,
            self.y_tf: self.y_test,
            self.t_tf: self.t_test,
            self.u_tf: self.u_test,
            self.v_tf: self.v_test,
        }

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 500 == 0 and hvd.rank() == 0:

                elapsed = time.time() - start_time
                pointsec = N_train * 500 / elapsed  # * hvd.size()
                loss_value = self.sess.run(self.loss, tf_dict)
                loss_value_test = self.sess.run(self.loss, tf_test_dict)

                lambda_1_value = self.sess.run(self.lambda_1)
                lambda_2_value = self.sess.run(self.lambda_2)

                error_lambda_1 = np.abs(lambda_1_value - 1.0) * 100
                error_lambda_2 = np.abs(lambda_2_value - 0.01) / 0.01 * 100

                u_pred, v_pred, p_pred = self.predict(
                    self.x_test, self.y_test, self.t_test
                )

                error_u = np.linalg.norm(self.u_test - u_pred, 2) / np.linalg.norm(
                    self.u_test, 2
                )
                error_v = np.linalg.norm(self.v_test - v_pred, 2) / np.linalg.norm(
                    self.v_test, 2
                )
                error_p = np.linalg.norm(self.p_test - p_pred, 2) / np.linalg.norm(
                    self.p_test, 2
                )

                print(error_u, error_v, error_p, "errors")
                print(
                    "It: %d, Loss: %.3e, Loss Test: %.3e, pps: %.3f, l1: %.3f, l2: %.5f, Time: %.2f"
                    % (
                        it,
                        loss_value,
                        loss_value_test,
                        pointsec,
                        error_lambda_1,
                        error_lambda_2,
                        elapsed,
                    )
                )

                train_list.append(loss_value)
                test_list.append(loss_value_test)
                metrics_u_list.append(error_u)
                metrics_v_list.append(error_v)
                metrics_p_list.append(error_p)
                time_list.append(elapsed)
                pointsec_list.append(pointsec)
                lambda_1_list.append(lambda_1_value)
                lambda_2_list.append(lambda_2_value)
                start_time = time.time()

        self.train_list = train_list
        self.test_list = test_list
        self.metrics_u_list = metrics_u_list
        self.metrics_v_list = metrics_v_list
        self.metrics_p_list = metrics_p_list
        self.time_list = time_list
        self.pointsec_list = pointsec_list
        self.lambda_1_list = lambda_1_list
        self.lambda_2_list = lambda_2_list

    def predict(self, x_star, y_star, t_star):

        tf_dict = {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star}

        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        p_star = self.sess.run(self.p_pred, tf_dict)

        return u_star, v_star, p_star


def plot_solution(X_star, u_star, index):

    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x, y)

    U_star = griddata(X_star, u_star.flatten(), (X, Y), method="cubic")

    plt.figure(index)
    plt.pcolor(X, Y, U_star, cmap="jet")
    plt.colorbar()


def axisEqual3D(ax):
    extents = np.array([getattr(ax, "get_{}lim".format(dim))() for dim in "xyz"])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 4
    for ctr, dim in zip(centers, "xyz"):
        getattr(ax, "set_{}lim".format(dim))(ctr - r, ctr + r)


print(N_train, "N")
print(epochs, "epochs")

layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]

# Load Data
data = scipy.io.loadmat("../PINNs/main/Data/cylinder_nektar_wake.mat")

U_star = data["U_star"]  # N x 2 x T
P_star = data["p_star"]  # N x T
t_star = data["t"]  # T x 1
X_star = data["X_star"]  # N x 2

N = X_star.shape[0]
T = t_star.shape[0]

# Rearrange Data
XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
TT = np.tile(t_star, (1, N)).T  # N x T

UU = U_star[:, 0, :]  # N x T
VV = U_star[:, 1, :]  # N x T
PP = P_star  # N x T

x = XX.flatten()[:, None]  # NT x 1
y = YY.flatten()[:, None]  # NT x 1
t = TT.flatten()[:, None]  # NT x 1

u = UU.flatten()[:, None]  # NT x 1
v = VV.flatten()[:, None]  # NT x 1
p = PP.flatten()[:, None]  # NT x 1

######################################################################
######################## Noiseles Data ###############################
######################################################################
# Training Data
idx = np.random.choice(N * T, N_train, replace=False)
x_train = x[idx, :]
y_train = y[idx, :]
t_train = t[idx, :]
u_train = u[idx, :]
v_train = v[idx, :]

# Testing Data
idx_test = np.random.choice(N * T, N_train, replace=False)
x_test = x[idx_test, :]
y_test = y[idx_test, :]
t_test = t[idx_test, :]

u_test = u[idx_test, :]
v_test = v[idx_test, :]
p_test = p[idx_test, :]


# Training

model = PhysicsInformedNN(
    x_train,
    y_train,
    t_train,
    u_train,
    v_train,
    x_test,
    y_test,
    t_test,
    u_test,
    v_test,
    p_test,
    layers,
)
start_time = time.time()
model.train(epochs)

if hvd.rank() == 0:
    elapsed = time.time() - start_time
    theta = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

    print("Training time: %.4f" % (elapsed))
    print(theta, "# Parameters:")
    print(N_train, "N")

    train_list = np.array(model.train_list)
    test_list = np.array(model.test_list)
    metrics_u_list = np.array(model.metrics_u_list)
    metrics_v_list = np.array(model.metrics_v_list)
    metrics_p_list = np.array(model.metrics_p_list)
    time_list = np.array(model.time_list)
    pointsec_list = np.array(model.pointsec_list)
    lambda_1_list = np.array(model.lambda_1_list)
    lambda_2_list = np.array(model.lambda_2_list)

    niter = time_list.shape[0]
    my_dict = {  #'hvd' : niter * [hvd.size()],
        "texec": elapsed,
        "theta": theta,
        "train": train_list,
        "test": test_list,
        "metrics_u": metrics_u_list,
        "metrics_v": metrics_v_list,
        "metrics_p": metrics_p_list,
        "times": time_list,
        "pointsec": pointsec_list,
        "lambda_1": lambda_1_list,
        "lambda_2": lambda_2_list,
        "date": now,
    }

    if args.save:
        np.save(name + ".npy", my_dict)

    # Test Data
    snap = np.array([100])
    x_star = X_star[:, 0:1]
    y_star = X_star[:, 1:2]
    t_star = TT[:, snap]

    u_star = U_star[:, 0, snap]
    v_star = U_star[:, 1, snap]
    p_star = P_star[:, snap]

    # Prediction
    u_pred, v_pred, p_pred = model.predict(x_star, y_star, t_star)
    lambda_1_value = model.sess.run(model.lambda_1)
    lambda_2_value = model.sess.run(model.lambda_2)

    # Error
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
    error_p = np.linalg.norm(p_star - p_pred, 2) / np.linalg.norm(p_star, 2)

    error_lambda_1 = np.abs(lambda_1_value - 1.0) * 100
    error_lambda_2 = np.abs(lambda_2_value - 0.01) / 0.01 * 100

    print("Error u: %e" % (error_u))
    print("Error v: %e" % (error_v))
    print("Error p: %e" % (error_p))
    print("Error l1: %.5f%%" % (error_lambda_1))
    print("Error l2: %.5f%%" % (error_lambda_2))
