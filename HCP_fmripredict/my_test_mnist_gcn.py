#!/home/yuzhang/tensorflow-py3.6/bin/python3.6

# Author: Yu Zhang
# License: simplified BSD
# coding: utf-8

import numpy as np
import time
import sys, os
sys.path.append('/home/yu/PycharmProjects/HCP_fmripredict/')
from cnn_graph.lib import models, graph, coarsening, utils

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

num_CPU = 2
config_TF = tf.ConfigProto(intra_op_parallelism_threads=num_CPU,\
        inter_op_parallelism_threads=num_CPU, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU})
session = tf.Session(config=config_TF)

flags = tf.app.flags
FLAGS = flags.FLAGS

# Graphs.
flags.DEFINE_integer('number_edges', 8, 'Graph: minimum number of edges per vertex.')
flags.DEFINE_string('metric', 'euclidean', 'Graph: similarity measure (between features).')
flags.DEFINE_bool('normalized_laplacian', True, 'Graph Laplacian: normalized.')
flags.DEFINE_integer('coarsening_levels', 4, 'Number of coarsened graphs.')
# Directories.
flags.DEFINE_string('dir_data', os.path.join('..', 'data', 'mnist'), 'Directory to store data.')

img_dim = 28


def grid_graph(m, corners=False):
    ##build a graph on minist digit images
    z = graph.grid(m)
    dist, idx = graph.distance_sklearn_metrics(z, k=FLAGS.number_edges, metric=FLAGS.metric)
    A = graph.adjacency(dist, idx)

    # Connections are only vertical or horizontal on the grid.
    # Corner vertices are connected to 2 neightbors only.
    if corners:
        import scipy.sparse
        A = A.toarray()
        A[A < A.max() / 1.5] = 0
        A = scipy.sparse.csr_matrix(A)
        print('{} edges'.format(A.nnz))

    ##plt.spy(A, markersize=2, color='black')
    print("{} > {} edges".format(A.nnz // 2, FLAGS.number_edges * m ** 2 // 2))
    return A

##data preparation
mnist = input_data.read_data_sets(FLAGS.dir_data, one_hot=False)

train_data = mnist.train.images.astype(np.float32)
val_data = mnist.validation.images.astype(np.float32)
test_data = mnist.test.images.astype(np.float32)
train_labels = mnist.train.labels
val_labels = mnist.validation.labels
test_labels = mnist.test.labels

###cal the adjcent matrix based on euclidean distance of spatial locations
A = grid_graph(img_dim, corners=False)
A = graph.replace_random_edges(A, 0.01)
###build multi-level graph using coarsen (div by 2 at each level)
graphs, perm = coarsening.coarsen(A, levels=FLAGS.coarsening_levels, self_connections=False)
L = [graph.laplacian(A, normalized=True) for A in graphs]

###paramters for the model
common = {}
common['dir_name']       = 'mnist/'
common['num_epochs']     = 20
common['batch_size']     = 100
common['decay_steps']    = mnist.train.num_examples / common['batch_size']
common['eval_frequency'] = 30 * common['num_epochs']
common['brelu']          = 'b1relu'
common['pool']           = 'mpool1'
C = max(mnist.train.labels) + 1  # number of classes

train_data_perm = coarsening.perm_data(train_data, perm)
val_data_perm = coarsening.perm_data(val_data, perm)
test_data_perm = coarsening.perm_data(test_data, perm)
model_perf = utils.model_perf()

###test different param settins
##model1: no convolution
name = 'softmax'
params = common.copy()
params['dir_name'] += name
params['regularization'] = 5e-4
params['dropout']        = 1
params['learning_rate']  = 0.02
params['decay_rate']     = 0.95
params['momentum']       = 0.9
params['F']              = []
params['K']              = []
params['p']              = []
params['M']              = [C]

####training and testing models
print(L)

t_start = time.process_time()
model_perf.test(models.cgcnn(config_TF, L, **params), name, params,
                train_data_perm, train_labels, val_data_perm, val_labels, test_data_perm, test_labels)
t_end_1 = time.process_time() - t_start
print('Model {}; Execution time: {:.2f}s\n\n'.format(name, t_end_1))

###model#2: one-layer convolution with fourier transform as filter
common['regularization'] = 0
common['dropout']        = 1
common['learning_rate']  = 0.02
common['decay_rate']     = 0.95
common['momentum']       = 0.9
common['F']              = [10]  # Number of graph convolutional filters.
common['K']              = [20]  # Polynomial orders.
common['p']              = [1]  # Pooling sizes.
common['M']              = [C] # Output dimensionality of fully connected layers.

name = 'fgconv_softmax'
params = common.copy()
params['dir_name'] += name
params['filter'] = 'fourier'
params['K'] = [L[0].shape[0]]

t_start = time.process_time()
model_perf.test(models.cgcnn(config_TF, L, **params), name, params,
                train_data_perm, train_labels, val_data_perm, val_labels, test_data_perm, test_labels)
t_end_2 = time.process_time() - t_start
print('Model {}; Execution time: {:.2f}s\n\n'.format(name, t_end_2))

##model#3: one-layer convolution with chebyshev5 and b1relu as filters
name = 'cgconv_softmax'
params = common.copy()
params['dir_name'] += name
params['filter'] = 'chebyshev5'
#    params['filter'] = 'chebyshev2'
#    params['brelu'] = 'b2relu'

t_start = time.process_time()
model_perf.test(models.cgcnn(config_TF,L, **params), name, params,
                train_data_perm, train_labels, val_data_perm, val_labels, test_data_perm, test_labels)
t_end_3 = time.process_time() - t_start
print('Model {}; Execution time: {:.2f}s\n\n'.format(name, t_end_3))

##model#4: two convolutional layers with fourier transform as filters
common['regularization'] = 5e-4
common['dropout']        = 0.5
common['learning_rate']  = 0.02  # 0.03 in the paper but sgconv_sgconv_fc_softmax has difficulty to converge
common['decay_rate']     = 0.95
common['momentum']       = 0.9
common['F']              = [32, 64]  # Number of graph convolutional filters.
common['K']              = [25, 25]  # Polynomial orders.
common['p']              = [4, 4]    # Pooling sizes.
common['M']              = [512, C]  # Output dimensionality of fully connected layers.

name = 'fgconv_fgconv_fc_softmax' #  'Non-Param'
params = common.copy()
params['dir_name'] += name
params['filter'] = 'fourier'
params['K'] = [L[0].shape[0], L[2].shape[0]]
print([L[li].shape for li in range(len(L))])

t_start = time.process_time()
model_perf.test(models.cgcnn(config_TF,L, **params), name, params,
                train_data_perm, train_labels, val_data_perm, val_labels, test_data_perm, test_labels)
t_end_4 = time.process_time() - t_start
print('Model {}; Execution time: {:.2f}s\n\n'.format(name, t_end_4))



##model#5: two convolutional layers with Chebyshev polynomial as filters
name = 'cgconv_cgconv_fc_softmax' #  'Non-Param'
params = common.copy()
params['dir_name'] += name
params['filter'] = 'chebyshev5'
print(params)
print([L[li].shape for li in range(len(L))])

t_start = time.process_time()
model_perf.test(models.cgcnn(config_TF,L, **params), name, params,
                train_data_perm, train_labels, val_data_perm, val_labels, test_data_perm, test_labels)
t_end_5 = time.process_time() - t_start
print('Model {}; Execution time: {:.2f}s\n\n'.format(name, t_end_5))



###summary
model_perf.show()
print('Execution time for model1: {:.2f}s\n\n'.format(t_end_1))
print('Execution time for model2: {:.2f}s\n\n'.format(t_end_2))
print('Execution time for model3: {:.2f}s\n\n'.format(t_end_3))
print('Execution time for model4: {:.2f}s\n\n'.format(t_end_4))
print('Execution time for model5: {:.2f}s\n\n'.format(t_end_5))
