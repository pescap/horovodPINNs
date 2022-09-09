import os
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
from mpi4py import MPI
import horovod.tensorflow as hvd
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser(description='Set parameters')

parser.add_argument("--seed", default=1234, type=int)
args = parser.parse_args()
seed = args.seed


np.random.seed(seed)
tf.set_random_seed(seed)
mode = 'uniform'
#mode = 'random'

hvd.init()
config = tf.ConfigProto()
config.gpu_options.visible_device_list = str(hvd.local_rank())
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

name = 'exp2/' + mode + '_seed' + str(seed) + '_size'+ str(hvd.size())


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

def data_uniform():
    if hvd.size() == 1:
        x_col = np.linspace(-1, 7, Ntot).reshape((-1, 1))
    if hvd.size() == 2:
        x_col = np.linspace(-1 + 4 * hvd.rank(), -1 + 4 * (hvd.rank() + 1), int(Ntot /2.)).reshape((-1, 1))
    if hvd.size() == 4:
        x_col = np.linspace(-1 + 2 * hvd.rank(), -1 + 2 * (hvd.rank() + 1), int(Ntot /4.)).reshape((-1, 1))
    if hvd.size() == 8:
        x_col = np.linspace(-1 + hvd.rank(), -1 + (hvd.rank() + 1), int(Ntot /8.)).reshape((-1, 1))
    return x_col
    
def data_random():
    if hvd.size() == 1:
        x_col = (np.random.random(Ntot) * 8 - 1).reshape((-1,1))
        
    if hvd.size() == 2:
        x_col = (np.random.random(int(Ntot/2)) * 8 - 1).reshape((-1,1))
    
    if hvd.size() == 4:
        x_col = (np.random.random(int(Ntot/4)) * 8 - 1).reshape((-1,1))
        
    if hvd.size() == 8:
        x_col = (np.random.random(int(Ntot/8)) * 8 - 1).reshape((-1,1))
    return x_col
        

print ('This is from rank %d'%(hvd.rank()))

N = 50 # Number of training data
Ntot = 8 * N
N_plot = 450
    

if mode == 'uniform':
    x_col = data_uniform()
elif mode == 'random':
    x_col = data_random()
else:
    print('error')


layers = [1] + 4*[50] + [1]
L = len(layers)
W = [hyper_initial([layers[l-1], layers[l]]) for l in range(1, L)] 
b = [tf.Variable(tf.zeros([1, layers[l]])) for l in range(1, L)] 

x_train = tf.placeholder(tf.float32, shape=[None, 1]) 
y_train = tf.placeholder(tf.float32, shape=[None, 1]) 
y_nn = DNN(x_train, W, b) 

loss = tf.reduce_mean(tf.square(y_nn - y_train)) 

lr = 0.001

optimizer=tf.train.AdamOptimizer(lr) 
optimizer=hvd.DistributedOptimizer(optimizer)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init) 

bcast = hvd.broadcast_global_variables(0)
sess.run(bcast)

y_col = np.sin(np.pi*x_col)
train_dict = {x_train: x_col, y_train: y_col}

Nmax = 10000 # Iteration counter

start_time = time.perf_counter()
n = 0
loss_list = []
while n <= Nmax:
    y_pred, train_, loss_ = sess.run([y_nn, train, loss], feed_dict=train_dict)
    if n%100 == 0 and hvd.rank() == 0:
        print('n = %d, loss = %.3e'%(n, loss_))
        loss_list.append(loss_)

    n += 1


if hvd.rank() == 0:
    
    
    stop_time = time.perf_counter()    
    texec = stop_time - start_time
    print('Rank: %d, Elapsed time: %f s'%(hvd.rank(), texec))

    xplot = np.linspace(-1, 7, N_plot).reshape((-1, 1)) 
    y_exact = np.sin(np.pi*xplot)
    y_pred_ = sess.run(y_nn, feed_dict={x_train: xplot})
    
    err_l2 = np.linalg.norm((y_pred_ - y_exact) / np.linalg.norm(y_exact))
    
    print(err_l2, 'L2-norm')
    
    loss_list = np.array(loss_list)
    
    my_dict = {'y_pred_' : y_pred_,
               'texec': texec,
               'err': err_l2,
               'loss': loss_list
               }
    
    np.save(name + '.npy',  my_dict)
    