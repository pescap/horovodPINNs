import os
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
from mpi4py import MPI
import horovod.tensorflow as hvd
import numpy as np
import time
import matplotlib.pyplot as plt

np.random.seed(1234)
tf.set_random_seed(1234)

#method = 'full'
method = 'parallel'


def hyper_initial(size):
    in_dim = size[0]
    out_dim = size[1]
    std = np.sqrt(2.0/(in_dim + out_dim))
    return tf.Variable(tf.random_normal(shape=size, stddev = std))

def DNN(X, W, b):
    A = X
    L = len(W)
    for i in range(L-1):
        A = tf.tanh(tf.add(tf.matmul(A, W[i]), b[i]))
    Y = tf.add(tf.matmul(A, W[-1]), b[-1])
    return Y

def pdenn(X, W, b):
    u = DNN(X, W, b)
    u_x = tf.gradients(u, X)[0]
    u_xx = tf.gradients(u_x, X)[0]
    f = 4*tf.sin(2*np.pi*X)*np.pi*np.pi

    R = u_xx + f

    return R

def data(N, method):
    if method == 'full':
        x_col = np.linspace(-1, hvd.size()-1, N * hvd.size()).reshape((-1, 1))
    if method == 'parallel':
        x_col = np.linspace(hvd.rank()-1, hvd.rank(), N).reshape((-1, 1))
    if method == 'ml':
        print('Error')
        x_col = np.linspace(hvd.rank()-1, hvd.rank(), N).reshape((-1, 1))
    return x_col
        
    
   
hvd.init()
config = tf.ConfigProto()
config.gpu_options.visible_device_list = str(hvd.local_rank())
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
#saver = tf.train.Saver()

print ('This is from rank %d'%(hvd.rank()))

N = 51 # Number of training data

x_col = data(N, method)

#x_col = np.linspace(-1, hvd.size(), N).reshape((-1, 1))
x_0 = np.array([-1]).reshape((-1, 1))
x_1 = np.array([1]).reshape((-1, 1))
y_0 = np.sin(2*np.pi*x_0)
y_1 = np.sin(2*np.pi*x_1)

layers = [1] + 3*[150] + [1]
L = len(layers)
W = [hyper_initial([layers[l-1], layers[l]]) for l in range(1, L)] 
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

loss = tf.reduce_mean(tf.square(R_nn)) + \
       tf.reduce_mean(tf.square(y_0_nn - y_0_train)) + \
       tf.reduce_mean(tf.square(y_1_nn - y_1_train)) 

#lr = 0.001 *hvd.size()
lr = 0.0001

optimizer=tf.train.AdamOptimizer(lr) 
optimizer=hvd.DistributedOptimizer(optimizer)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init) 

bcast = hvd.broadcast_global_variables(0)
sess.run(bcast)

train_dict = {x_train: x_col, x_0_train: x_0, y_0_train: y_0, x_1_train: x_1, y_1_train: y_1}

Nmax = 5000 # Iteration counter

start_time = time.perf_counter()
n = 0
loss_list = []
while n <= Nmax:
    y_pred, train_, loss_ = sess.run([y_nn, train, loss], feed_dict=train_dict)
    n += 1
    if n%100 == 0 and hvd.rank() == 0:
        print('n = %d, loss = %.3e'%(n, loss_))
    if hvd.rank() == 0:
        loss_list.append(loss_)

if hvd.rank() == 0:
    stop_time = time.perf_counter()
    print('Rank: %d, Elapsed time: %f s'%(hvd.rank(), stop_time - start_time))

    N_plot = 450
    xplot = np.linspace(-1, hvd.size()-1, N_plot).reshape((-1, 1)) 
    y_exact = np.sin(2*np.pi*xplot)

    y_pred_ = sess.run(y_nn, feed_dict={x_train: xplot})

    filename = 'pinn_p' + str(N) + str(hvd.size())+ str(Nmax)
    np.savetxt('results/y_pred_' + filename, y_pred_, fmt='%e')
    np.savetxt('results/loss_' + filename, loss_list, fmt='%e')

    from matplotlib import pyplot as plt
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize = (12,6))

    for rank in range(hvd.size()):
        x_col = np.linspace(rank-1, rank, N).reshape((-1, 1))
        y_col = sess.run(y_nn, feed_dict={x_train: x_col})
        ax1.plot(x_col, 0 * x_col, 'o')
        ax1.plot(x_col, y_col, 'o-', color = 'r')
        ax1.plot(xplot,y_exact, color = 'g')                         

    ax2.plot(xplot,y_pred_, color = 'r', label = 'PINN')
    ax2.plot(xplot,y_exact, color = 'g', label = 'exact')

    ax1.legend()
    ax2.legend()
    plt.savefig('plots/' + filename)        