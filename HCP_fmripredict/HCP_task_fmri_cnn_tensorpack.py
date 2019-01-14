#!/home/yuzhang/tensorflow-py3.6/bin/python3.6

# Author: Yu Zhang
# License: simplified BSD
# coding: utf-8

from pathlib import Path
import glob
import itertools
import os
import time
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
###%matplotlib inline

from nilearn import signal
from nilearn import image
from sklearn import preprocessing
from keras.utils import np_utils

from tensorpack import dataflow

from keras.utils import to_categorical
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout,AveragePooling2D
from keras.layers import Conv3D, MaxPooling3D, BatchNormalization, AveragePooling3D
from keras.models import Model
import keras.backend as K


#####global variable settings
'''
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
'''
import tensorflow as tf
from keras import backend as K

USE_GPU_CPU = 1
num_cores = 4

if not USE_GPU_CPU :
    num_GPU = num_cores
    num_CPU = 0
else:
    num_CPU = 2
    num_GPU = 2

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = tf.Session(config=config)
K.set_session(session)


#########################################################
pathdata = Path('/project/6002071/yuzhang/HCP/aws_s3_HCP1200/FMRI/')
pathout = '/project/6002071/yuzhang/HCP/'
modality = 'MOTOR'  # 'MOTOR'
###dict for different types of movement
task_contrasts = {"rf": "foot",
                  "lf": "foot",
                  "rh": "hand",
                  "lh": "hand",
                  "t": "tongue"}
target_name = np.unique(list(task_contrasts.values()))
print(target_name)

TR = 0.72
nr_thread=5
buffer_size=20
Flag_CNN_Model = '2d'
########################


def load_fmri_data(pathdata,modality=None,confound_name=None):
    ###fMRI decoding: using event signals instead of activation pattern from glm
    ##collect task-fMRI signals

    if not modality:
        modality = 'MOTOR'  # 'MOTOR'

    subjects = []
    fmri_files = []
    confound_files = []
    for fmri_file in sorted(pathdata.glob('tfMRI_'+modality+'_??/*_tfMRI_'+modality+'_??.nii.gz')):
        subjects.append(Path(os.path.dirname(fmri_file)).parts[-3])
        fmri_files.append(str(fmri_file))

    for confound in sorted(pathdata.glob('tfMRI_'+modality+'_??/*_Movement_Regressors.txt')):
        confound_files.append(str(confound))

    print('%d subjects included in the dataset' % len(fmri_files))
    return fmri_files, confound_files, subjects


def load_event_files(fmri_files,confound_files,ev_filename=None):
    ###collect the event design files
    tc_matrix = nib.load(fmri_files[0])
    Subject_Num = len(fmri_files)
    Trial_Num = tc_matrix.shape[-1]
    print("Data samples including %d subjects with %d trials" % (Subject_Num, Trial_Num))

    EVS_files = []
    subj = 0
    for ev, sub_count in zip(sorted(pathdata.glob('tfMRI_' + modality + '_??/*_combined_events_spm_' + modality + '.csv')),range(Subject_Num)):
        ###remove fmri files if the event design is missing
        while os.path.dirname(fmri_files[subj]) < os.path.dirname(str(ev)):
            print("Event files and fmri data are miss-matching for subject: ")
            print(Path(os.path.dirname(str(ev))).parts[-3::2], ':',
                  Path(os.path.dirname(fmri_files[subj])).parts[-3::2])
            print("Due to missing event files for subject : %s" % os.path.dirname(fmri_files[subj]))
            fmri_files[subj] = []
            confound_files[subj] = []
            subj += 1
            if subj > Subject_Num:
                break
        if os.path.dirname(fmri_files[subj]) == os.path.dirname(str(ev)):
            EVS_files.append(str(ev))
            subj += 1

    fmri_files = list(filter(None, fmri_files))
    confound_files = list(filter(None, confound_files))
    if len(EVS_files) != len(fmri_files):
        print('Miss-matching number of subjects between event:{} and fmri:{} files'.format(len(EVS_files), len(fmri_files)))

    ################################
    ###loading all event designs
    if not ev_filename:
        ev_filename = "_event_labels_1200R_LR_RL.txt"

    events_all_subjects_file = pathout+modality+ev_filename
    if os.path.isfile(events_all_subjects_file):
        print('Collecting trial info from file:', events_all_subjects_file)
        subjects_trial_labels = pd.read_csv(events_all_subjects_file,sep="\t",encoding="utf8")
        ###print(subjects_trial_labels.keys())

        subjects_trial_label_matrix = subjects_trial_labels.loc[:,'trial1':'trial'+str(Trial_Num)]
        sub_name = subjects_trial_labels['subject']
        coding_direct = subjects_trial_labels['coding']
        print(subjects_trial_label_matrix.shape,len(sub_name),len(np.unique(sub_name)),len(coding_direct))
    else:
        print('Loading trial info for each task-fmri file and save to csv file:', events_all_subjects_file)
        subjects_trial_label_matrix = []
        sub_name = []
        coding_direct = []
        for subj in np.arange(Subject_Num):
            pathsub = Path(os.path.dirname(EVS_files[subj]))
            sub_name.append(pathsub.parts[-3])
            coding_direct.append(pathsub.parts[-1].split('_')[-1])

            ##trial info in volume
            trial_infos = pd.read_csv(EVS_files[subj],sep="\t",encoding="utf8",header = None,names=['onset','duration','rep','task'])
            Onsets = np.ceil((trial_infos.onset/TR)).astype(int) #(trial_infos.onset/TR).astype(int)
            Duras = np.ceil((trial_infos.duration/TR)).astype(int) #(trial_infos.duration/TR).astype(int)
            Movetypes = trial_infos.task

            labels = ["rest"]*Trial_Num;
            for start,dur,move in zip(Onsets,Duras,Movetypes):
                for ti in range(start-1,start+dur):
                    labels[ti]= task_contrasts[move]
            subjects_trial_label_matrix.append(labels)

        print(np.array(subjects_trial_label_matrix).shape)
        #print(np.array(subjects_trial_label_matrix[0]))
        subjects_trial_labels = pd.DataFrame(data=np.array(subjects_trial_label_matrix),columns=['trial'+str(i+1) for i in range(Trial_Num)])
        subjects_trial_labels['subject'] = sub_name
        subjects_trial_labels['coding'] = coding_direct
        subjects_trial_labels.keys()
        #print(subjects_trial_labels['subject'],subjects_trial_labels['coding'])

        ##save the labels
        subjects_trial_labels.to_csv(events_all_subjects_file,sep='\t', encoding='utf-8',index=False)

    return subjects_trial_label_matrix, sub_name


#############################
#######################################
####tensorpack: multithread
class gen_fmri_file(dataflow.DataFlow):
    """ Iterate through fmri filenames, confound filenames and labels
    """
    def __init__(self, fmri_files,confound_files, label_matrix,data_type='train',train_percent=0.8):
        assert (len(fmri_files) == len(confound_files))
        # self.data=zip(fmri_files,confound_files)
        self.fmri_files = fmri_files
        self.confound_files = confound_files
        self.label_matrix = label_matrix

        self.data_type=data_type
        self.train_percent=train_percent

    def size(self):
        split_num=int(len(self.fmri_files)*0.8)
        if self.data_type=='train':
            return split_num
        else:
            return len(self.fmri_files)-split_num

    def get_data(self):
        split_num=int(len(self.fmri_files)*0.8)
        if self.data_type=='train':
            while True:
                rand_pos=np.random.choice(split_num,1)[0]
                yield self.fmri_files[rand_pos],self.confound_files[rand_pos],self.label_matrix.iloc[rand_pos]
        else:
            for pos_ in range(split_num,len(self.fmri_files)):
                yield self.fmri_files[pos_],self.confound_files[pos_],self.label_matrix.iloc[pos_]


class split_samples(dataflow.DataFlow):
    """ Iterate through fmri filenames, confound filenames and labels
    """
    def __init__(self, ds):
        self.ds=ds

    def size(self):
        return 91*284

    def get_data(self):
        for data in self.ds.get_data():
            for i in range(data[1].shape[0]):
                yield data[0][i],data[1][i]


def map_load_fmri_image(dp,target_name):
    fmri_file=dp[0]
    confound_file=dp[1]
    label_trials=dp[2]

    ###remove confound effects
    confound = np.loadtxt(confound_file)
    fmri_data_clean = image.clean_img(fmri_file, detrend=True, standardize=True, confounds=confound)

    ##pre-select task types
    trial_mask = pd.Series(label_trials).isin(target_name)  ##['hand', 'foot','tongue']
    fmri_data_cnn = image.index_img(fmri_data_clean, np.where(trial_mask)[0]).get_data()
    ###use each slice along z-axis as one sample
    label_data_trial = np.array(label_trials.loc[trial_mask])
    le = preprocessing.LabelEncoder()
    le.fit(target_name)
    label_data_cnn = le.transform(label_data_trial) ##np_utils.to_categorical(): convert label vector to matrix

    img_rows, img_cols, img_deps = fmri_data_cnn.shape[:-1]
    fmri_data_cnn_test = np.transpose(fmri_data_cnn.reshape(img_rows, img_cols, np.prod(fmri_data_cnn.shape[2:])), (2, 0, 1))
    label_data_cnn_test = np.repeat(label_data_cnn, img_deps, axis=0).flatten()
    print(fmri_file, fmri_data_cnn_test.shape,label_data_cnn_test.shape)

    return fmri_data_cnn_test, label_data_cnn_test


def map_load_fmri_image_3d(dp, target_name):
    fmri_file = dp[0]
    confound_file = dp[1]
    label_trials = dp[2]

    ###remove confound effects
    confound = np.loadtxt(confound_files[0])
    fmri_data_clean = image.clean_img(fmri_files[0], detrend=True, standardize=True, confounds=confound)

    ##pre-select task types
    trial_mask = pd.Series(label_trials).isin(target_name)  ##['hand', 'foot','tongue']
    fmri_data_cnn = image.index_img(fmri_data_clean, np.where(trial_mask)[0]).get_data()
    ###use each slice along z-axis as one sample
    label_data_trial = np.array(label_trials.loc[trial_mask])
    le = preprocessing.LabelEncoder()
    le.fit(target_name)
    label_data_cnn = le.transform(label_data_trial)  ##np_utils.to_categorical(): convert label vector to matrix

    img_rows, img_cols, img_deps = fmri_data_cnn.shape[:-1]
    fmri_data_cnn_test = np.transpose(fmri_data_cnn, (3, 0, 1, 2))
    label_data_cnn_test = label_data_cnn.flatten()
    print(fmri_file, fmri_data_cnn_test.shape, label_data_cnn_test.shape)

    return fmri_data_cnn_test, label_data_cnn_test


def data_pipe(fmri_files,confound_files,label_matrix,target_name=None,batch_size=32,data_type='train',
              train_percent=0.8,nr_thread=nr_thread,buffer_size=buffer_size):
    assert data_type in ['train', 'val', 'test']
    assert fmri_files is not None

    print('\n\nGenerating dataflow for %s datasets \n' % data_type)

    buffer_size = min(len(fmri_files),buffer_size)
    nr_thread = min(len(fmri_files),nr_thread)

    ds0 = gen_fmri_file(fmri_files,confound_files, label_matrix,data_type=data_type,train_percent=train_percent)
    print('dataflowSize is ' + str(ds0.size()))
    print('Loading data using %d threads with %d buffer_size ... \n' % (nr_thread, buffer_size))

    if target_name is None:
        target_name = np.unique(label_matrix)

    ####running the model
    start_time = time.clock()
    ds1 = dataflow.MultiThreadMapData(
        ds0, nr_thread=nr_thread,
        map_func=lambda dp: map_load_fmri_image(dp,target_name),
        buffer_size=buffer_size,
        strict=True)

    ds1 = dataflow.PrefetchData(ds1, buffer_size,1)

    ds1 = split_samples(ds1)
    print('prefetch dataflowSize is ' + str(ds1.size()))

    ds1 = dataflow.LocallyShuffleData(ds1,buffer_size=ds1.size()*buffer_size)

    ds1 = dataflow.BatchData(ds1,batch_size=batch_size)
    print('Time Usage of loading data in seconds: {} \n'.format(time.clock() - start_time))

    ds1 = dataflow.PrefetchDataZMQ(ds1, nr_proc=1)
    ds1._reset_once()
    ##ds1.reset_state()

    #return ds1.get_data()
    for df in ds1.get_data():
        ##print(np.expand_dims(df[0].astype('float32'),axis=3).shape)
        yield (np.expand_dims(df[0].astype('float32'),axis=3),to_categorical(df[1].astype('int32'),len(target_name)))


def data_pipe_3dcnn(fmri_files, confound_files, label_matrix, target_name=None, flag_cnn='3d', batch_size=32,
                    data_type='train',train_percent=0.8, nr_thread=nr_thread, buffer_size=buffer_size):
    assert data_type in ['train', 'val', 'test']
    assert flag_cnn in ['3d', '2d']
    assert fmri_files is not None

    print('\n\nGenerating dataflow for %s datasets \n' % data_type)

    buffer_size = min(len(fmri_files), buffer_size)
    nr_thread = min(len(fmri_files), nr_thread)

    ds0 = gen_fmri_file(fmri_files, confound_files, label_matrix, data_type=data_type, train_percent=train_percent)
    print('dataflowSize is ' + str(ds0.size()))
    print('Loading data using %d threads with %d buffer_size ... \n' % (nr_thread, buffer_size))

    if target_name is None:
        target_name = np.unique(label_matrix)

    ####running the model
    start_time = time.clock()
    if flag_cnn == '2d':
        ds1 = dataflow.MultiThreadMapData(
            ds0, nr_thread=nr_thread,
            map_func=lambda dp: map_load_fmri_image(dp, target_name),
            buffer_size=buffer_size,
            strict=True)
    elif flag_cnn == '3d':
        ds1 = dataflow.MultiThreadMapData(
            ds0, nr_thread=nr_thread,
            map_func=lambda dp: map_load_fmri_image_3d(dp, target_name),
            buffer_size=buffer_size,
            strict=True)

    ds1 = dataflow.PrefetchData(ds1, buffer_size, 1)

    ds1 = split_samples(ds1)
    print('prefetch dataflowSize is ' + str(ds1.size()))

    ds1 = dataflow.LocallyShuffleData(ds1, buffer_size=ds1.size() * buffer_size)

    ds1 = dataflow.BatchData(ds1, batch_size=batch_size)
    print('Time Usage of loading data in seconds: {} \n'.format(time.clock() - start_time))

    ds1 = dataflow.PrefetchDataZMQ(ds1, nr_proc=1)
    ds1._reset_once()
    ##ds1.reset_state()

    ##return ds1.get_data()

    for df in ds1.get_data():
        if flag_cnn == '2d':
            yield (np.expand_dims(df[0].astype('float32'), axis=3),to_categorical(df[1].astype('int32'), len(target_name)))
        elif flag_cnn == '3d':
            yield (np.expand_dims(df[0].astype('float32'), axis=4),to_categorical(df[1].astype('int32'), len(target_name)))


###end of tensorpack: multithread
##############################################################


def plot_history(model_history):
    plt.figure()
    plt.subplot(121)
    plt.plot(model_history.history['acc'], color='r')
    plt.plot(model_history.history['val_acc'], color='b')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Training', 'Validation'])

    plt.subplot(122)
    plt.plot(model_history.history['loss'], color='r')
    plt.plot(model_history.history['val_loss'], color='b')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Function')
    plt.legend(['Training', 'Validation'])
    return None


def build_cnn_model(input_shape, Nlabels, filters=32, convsize=3, poolsize=2, hidden_size=128, conv_layers=4):
    #     import keras.backend as K
    #     if K.image_data_format() == 'channels_first':
    #         img_shape = (1,img_rows,img_cols)
    #     elif K.image_data_format() == 'channels_last':
    #         img_shape = (img_rows,img_cols,1)


    input0 = Input(shape=input_shape)
    drop1 = input0
    for li in range(conv_layers):
        conv1 = Conv2D(filters, (convsize, convsize), padding='same', activation='relu')(drop1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(filters, (convsize, convsize), padding='same', activation='relu')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D((poolsize, poolsize))(conv1)
        drop1 = Dropout(0.25)(pool1)
        if (li+1) % 2 == 0:
            filters *= 2

    drop2 = drop1
    avg1 = AveragePooling2D(pool_size=(5, 5))(drop2)
    flat = Flatten()(avg1)
    hidden = Dense(hidden_size, activation='relu')(flat)
    drop3 = Dropout(0.5)(hidden)
    # hidden = Dense((hidden_size/4).astype(int), activation='relu')(drop3)
    # drop4 = Dropout(0.5)(hidden)

    out = Dense(Nlabels, activation='softmax')(drop3)

    model = Model(inputs=input0, outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


def build_cnn3d_model(input_shape, Nlabels, filters=32, convsize=3, poolsize=2, hidden_size=128, conv_layers=4):
    #     import keras.backend as K
    #     if K.image_data_format() == 'channels_first':
    #         img_shape = (1,img_rows,img_cols,img_deps)
    #     elif K.image_data_format() == 'channels_last':
    #         img_shape = (img_rows,img_cols, img_deps,1)

    input0 = Input(shape=input_shape)
    drop1 = input0
    for li in range(conv_layers):
        conv1 = Conv3D(filters, (convsize, convsize, convsize), padding='same', activation='relu')(drop1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv3D(filters, (convsize, convsize, convsize), padding='same', activation='relu')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling3D((poolsize, poolsize, poolsize))(conv1)
        drop1 = Dropout(0.25)(pool1)
        if (li+1) % 2 == 0:
            filters *= 2

    drop2 = drop1
    avg1 = AveragePooling3D(pool_size=(5, 5, 5))(drop2)
    flat = Flatten()(avg1)
    hidden = Dense(hidden_size, activation='relu')(flat)
    drop3 = Dropout(0.5)(hidden)

    out = Dense(Nlabels, activation='softmax')(drop3)

    model = Model(inputs=input0, outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


#####################
#####
if __name__ == '__main__':

    fmri_files, confound_files, subjects = load_fmri_data(pathdata,modality)
    print('including %d fmri files and %d confounds files \n\n' % (len(fmri_files), len(confound_files)))

    label_matrix, sub_name = load_event_files(fmri_files,confound_files)
    print('Collecting event design files for subjects and saved into matrix ...' , np.array(label_matrix).shape)

    nb_class = len(target_name)
    tc_matrix = nib.load(fmri_files[0])
    img_rows, img_cols, img_deps = tc_matrix.shape[:-1]
    img_shape = []
    if Flag_CNN_Model == '2d':
        if K.image_data_format() == 'channels_first':
            img_shape = (1, img_rows, img_cols)
        elif K.image_data_format() == 'channels_last':
            img_shape = (img_rows, img_cols, 1)
    elif Flag_CNN_Model == '3d':
        if K.image_data_format() == 'channels_first':
            img_shape = (1, img_rows, img_cols, img_deps)
        elif K.image_data_format() == 'channels_last':
            img_shape = (img_rows, img_cols, img_deps, 1)

    #########################################
    '''
    ##test whether dataflow from tensorpack works
    test_sub_num = 1000
    tst = data_pipe_3dcnn(fmri_files[:test_sub_num], confound_files[:test_sub_num], label_matrix.iloc[:test_sub_num],
                          target_name=target_name, flag_cnn=Flag_CNN_Model, batch_size=16, data_type='train', buffer_size=5)
    out = next(tst)
    print(out[0].shape)
    print(out[1].shape)
    '''
    ####################
    #####start 2dcnn model
    test_sub_num = len(fmri_files)
    ##xx = data_pipe(fmri_files,confound_files,label_matrix,target_name=target_name)
    train_gen = data_pipe(fmri_files[:test_sub_num],confound_files[:test_sub_num],label_matrix.iloc[:test_sub_num],
                          target_name=target_name,batch_size=32,data_type='train',nr_thread=4, buffer_size=20)
    val_set = data_pipe(fmri_files[:test_sub_num],confound_files[:test_sub_num],label_matrix.iloc[:test_sub_num],
                        target_name=target_name,batch_size=32,data_type='test',nr_thread=2, buffer_size=20)

    '''
    #########################################
    test_sub_num = len(fmri_files)
    ######start cnn model
    train_gen = data_pipe_3dcnn(fmri_files[:test_sub_num], confound_files[:test_sub_num],label_matrix.iloc[:test_sub_num],
                                target_name=target_name, flag_cnn=Flag_CNN_Model,
                                batch_size=32, data_type='train', nr_thread=4, buffer_size=20)
    val_set = data_pipe_3dcnn(fmri_files[:test_sub_num], confound_files[:test_sub_num],label_matrix.iloc[:test_sub_num],
                              target_name=target_name, flag_cnn=Flag_CNN_Model,
                              batch_size=32, data_type='test', nr_thread=2, buffer_size=20)
    '''

    if Flag_CNN_Model == '2d':
        print('\nTraining the model using 2d-CNN \n')
        model_test = build_cnn_model(img_shape, nb_class)
    elif Flag_CNN_Model == '3d':
        print('\nTraining the model using 3d-CNN \n')
        model_test = build_cnn3d_model(img_shape, nb_class)

    ######start training the model
    model_test_history = model_test.fit_generator(train_gen, epochs=20, steps_per_epoch=100, verbose=1, shuffle=True)
                                                  #validation_data=val_set,validation_steps=10,
                                                  #workers=1, use_multiprocessing=False, shuffle=True)
    print(model_test_history.history)
    for key,val in model_test_history.history.items():
        print(key, val)

    scores = model_test.evaluate_generator(val_set, validation_steps=100, workers=1, shuffle=False)
    print(scores)

    import pickle
    logfilename = pathout+'train_val_scores_dump2.txt'
    if os.path.isfile(logfilename):
        logfilename = logfilename.split('.')[0] + '2.txt'
    file = open(logfilename, 'w')
    pickle.dump(model_test_history.history, file)
    file.close()

'''
from tensorpack import *
from tensorpack.tfutils import summary
from tensorpack.dataflow import dataset

class Model(ModelDesc):
    def inputs(self,image_shape):
        """
        Define all the inputs (with type, shape, name) that the graph will need.
        """
        return [tf.placeholder(tf.float32, (None, image_shape.rval()), 'input'),
                tf.placeholder(tf.int32, (None,), 'label')]

    def build_graph(self, image, label):
        """This function should build the model which takes the input variables
        and return cost at the end"""

        # In tensorflow, inputs to convolution function are assumed to be
        # NHWC. Add a single channel here.
        image = tf.expand_dims(image, 3)

        image = image * 2 - 1   # center the pixels values at zero
        # The context manager `argscope` sets the default option for all the layers under
        # this context. Here we use 32 channel convolution with shape 3x3
        with argscope(Conv2D, kernel_size=3, activation=tf.nn.relu, filters=32):
            logits = (LinearWrap(image)
                      .Conv2D('conv0')
                      .MaxPooling('pool0', 2)
                      .Conv2D('conv1')
                      .Conv2D('conv2')
                      .MaxPooling('pool1', 2)
                      .Conv2D('conv3')
                      .FullyConnected('fc0', 512, activation=tf.nn.relu)
                      .Dropout('dropout', rate=0.5)
                      .FullyConnected('fc1', 10, activation=tf.identity)())

        tf.nn.softmax(logits, name='prob')   # a Bx10 with probabilities

        # a vector of length B with loss of each sample
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')  # the average cross-entropy loss

        correct = tf.cast(tf.nn.in_top_k(logits, label, 1), tf.float32, name='correct')
        accuracy = tf.reduce_mean(correct, name='accuracy')

        # This will monitor training error (in a moving_average fashion):
        # 1. write the value to tensosrboard
        # 2. write the value to stat.json
        # 3. print the value after each epoch
        train_error = tf.reduce_mean(1 - correct, name='train_error')
        summary.add_moving_summary(train_error, accuracy)

        # Use a regex to find parameters to apply weight decay.
        # Here we apply a weight decay on all W (weight matrix) of all fc layers
        wd_cost = tf.multiply(1e-5,
                              regularize_cost('fc.*/W', tf.nn.l2_loss),
                              name='regularize_loss')
        total_cost = tf.add_n([wd_cost, cost], name='total_cost')
        summary.add_moving_summary(cost, wd_cost, total_cost)

        # monitor histogram of all weight (of conv and fc layers) in tensorboard
        summary.add_param_summary(('.*/W', ['histogram', 'rms']))
        return total_cost

def get_config(dataset_train,dataset_test):
    # How many iterations you want in each epoch.
    # This is the default value, don't actually need to set it in the config
    steps_per_epoch = dataset_train.size()

    # get the config which contains everything necessary in a training
    return TrainConfig(
        model=Model(),
        dataflow=dataset_train,  # the DataFlow instance for training
        callbacks=[
            ModelSaver(),   # save the model after every epoch
            MaxSaver('validation_accuracy'),  # save the model with highest accuracy (prefix 'validation_')
            InferenceRunner(    # run inference(for validation) after every epoch
                dataset_test,   # the DataFlow instance used for validation
                ScalarStats(['cross_entropy_loss', 'accuracy'])),
        ],
        steps_per_epoch=steps_per_epoch,
        max_epoch=100,
    )

##main function
config = get_config()
launch_train_with_config(config, SimpleTrainer())
'''
