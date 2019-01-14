#!/home/yuzhang/tensorflow-py3.6/bin/python3.6

# Author: Yu Zhang
# License: simplified BSD
# coding: utf-8

from pathlib import Path
import glob
import itertools
import lmdb
import h5py
import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
###%matplotlib inline

from nilearn import signal
from nilearn import image
from sklearn import preprocessing
from keras.utils import np_utils

import tensorflow as tf
from tensorpack import dataflow

from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.utils import to_categorical
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout,AveragePooling2D
from keras.layers import Conv3D, MaxPooling3D, BatchNormalization, AveragePooling3D
from keras.models import Model
from keras import regularizers
from keras import backend as K


#####global variable settings
'''
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
'''

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
##pathdata = Path('/home/yuzhang/scratch/HCP/aws_s3_HCP1200/FMRI/')
##pathout = '/home/yuzhang/scratch/HCP/'
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
nr_thread=4
buffer_size=20
test_size = 0.2
img_resize = 3
Flag_Block_Trial = 0
block_dura = 1

learning_rate = 0.001
learning_rate_decay = 0.9
l2_reg = 0.0001
Flag_CNN_Model = '2d'

###change buffer_size to save time
if Flag_CNN_Model == '2d':
    buffer_size=20
elif Flag_CNN_Model == '3d':
    buffer_size=40
########################


def load_fmri_data(pathdata,modality=None,confound_name=None):
    ###fMRI decoding: using event signals instead of activation pattern from glm
    ##collect task-fMRI signals

    if not modality:
        modality = 'MOTOR'  # 'MOTOR'

    subjects = []
    fmri_files = []
    confound_files = []
    for fmri_file in sorted(pathdata.glob('tfMRI_'+modality+'_??/*tfMRI_'+modality+'_??.nii.gz')):
        subjects.append(Path(os.path.dirname(fmri_file)).parts[-3])
        fmri_files.append(str(fmri_file))

    for confound in sorted(pathdata.glob('tfMRI_'+modality+'_??/*Movement_Regressors.txt')):
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
    for ev, sub_count in zip(sorted(pathdata.glob('tfMRI_' + modality + '_??/*combined_events_spm_' + modality + '.csv')),range(Subject_Num)):
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
        trial_infos = pd.read_csv(EVS_files[0],sep="\t",encoding="utf8",header = None,names=['onset','duration','rep','task'])
        Duras = np.ceil((trial_infos.duration/TR)).astype(int) #(trial_infos.duration/TR).astype(int)

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

    block_dura = np.unique(Duras)[0]
    return subjects_trial_label_matrix, sub_name, block_dura


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
	    return 1e20
        #split_num=int(len(self.fmri_files)*0.8)
        #if self.data_type=='train':
        #    return split_num
        #else:
        #    return len(self.fmri_files)-split_num

    def get_data(self):
        split_num=int(len(self.fmri_files) * self.train_percent)
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
        #return 91*284
        return 1e20

    def data_info(self):
        for data in self.ds.get_data():
            print('fmri/label data shape:',data[0].shape,data[1].shape)
            return data[0].shape

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
    try:
        fmri_data_clean = image.clean_img(fmri_file, detrend=True, standardize=True, confounds=confound)
    except:
        print('Error loading fmri file. Check fmri data first: %s' % fmri_file)
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
    ##print(fmri_file, fmri_data_cnn_test.shape,label_data_cnn_test.shape)

    return fmri_data_cnn_test, label_data_cnn_test

def map_load_fmri_image_3d(dp, target_name):
    fmri_file = dp[0]
    confound_file = dp[1]
    label_trials = dp[2]

    ###remove confound effects
    confound = np.loadtxt(confound_file)
    try:
        fmri_data_clean = image.clean_img(fmri_file, detrend=True, standardize=True, confounds=confound)
    except:
        print('Error loading fmri file. Check fmri data first: %s' % fmri_file)
        fmri_data_clean = image.clean_img(fmri_file, detrend=True, standardize=True, confounds=confound)

    ##resample image into a smaller size to save memory
    target_affine = np.diag((img_resize, img_resize, img_resize))
    fmri_data_clean_resample = image.resample_img(fmri_data_clean, target_affine=target_affine)
    ##print(fmri_data_clean_resample.get_data().shape)

    ##pre-select task types
    trial_mask = pd.Series(label_trials).isin(target_name)  ##['hand', 'foot','tongue']
    fmri_data_cnn = image.index_img(fmri_data_clean_resample, np.where(trial_mask)[0]).get_data()
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
    Trial_Num = ds1.data_info()[0]
    print('%d #Trials/Samples per subject' % Trial_Num)

    ##ds1 = dataflow.LocallyShuffleData(ds1,buffer_size=ds1.size()*buffer_size)
    ds1 = dataflow.LocallyShuffleData(ds1,buffer_size=Trial_Num * buffer_size)

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
    ##Subject_Num, Trial_Num = np.array(label_matrix).shape

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
    Trial_Num = ds1.data_info()[0]
    print('%d #Trials/Samples per subject' % Trial_Num)

    #ds1 = dataflow.LocallyShuffleData(ds1, buffer_size=ds1.size() * buffer_size)
    ds1 = dataflow.LocallyShuffleData(ds1, buffer_size=Trial_Num * buffer_size)

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


#################################################################
#########reshape fmri and label data into blocks
#######change: use blocks instead of trials as input
def map_load_fmri_image_block(dp,target_name,block_dura=1):
    ###extract time-series within each block in terms of trial numbers
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
    img_rows, img_cols, img_deps = fmri_data_cnn.shape[:-1]

    ##cut the trials into blocks
    chunks = int(np.floor(len(label_data_trial)/block_dura))
    fmri_data_cnn_block = np.array(np.array_split(fmri_data_cnn,chunks,axis=3))
    if block_dura != fmri_data_cnn_block.shape[-1]:
        print('adjust the duration of each block from %d to %d' % (block_dura,fmri_data_cnn_block.shape[-1]))
        block_dura = fmri_data_cnn_block.shape[-1]

    label_data_trial_block = np.array_split(label_data_trial,chunks)
    label_data_trial_block = list(zip(*label_data_trial_block))[0]
    le = preprocessing.LabelEncoder()
    le.fit(target_name)
    label_data_cnn_block = le.transform(label_data_trial_block) ##np_utils.to_categorical(): convert label vector to matrix

    ###reshape data to fit the model
    fmri_data_cnn_block = np.transpose(fmri_data_cnn_block,(3,0,1,2,4))
    fmri_data_cnn_test = fmri_data_cnn_block.reshape(np.prod(fmri_data_cnn_block.shape[:2]),img_rows, img_cols, block_dura)
    label_data_cnn_test = np.repeat(label_data_cnn_block, img_deps, axis=0).flatten()
    ##print(fmri_file, fmri_data_cnn_test.shape,label_data_cnn_test.shape)

    return fmri_data_cnn_test, label_data_cnn_test

def map_load_fmri_image_3d_block(dp, target_name,block_dura=1):
    fmri_file = dp[0]
    confound_file = dp[1]
    label_trials = dp[2]

    ###remove confound effects
    confound = np.loadtxt(confound_file)
    fmri_data_clean = image.clean_img(fmri_file, detrend=True, standardize=True, confounds=confound)
    ##resample image into a smaller size to save memory
    target_affine = np.diag((img_resize, img_resize, img_resize))
    fmri_data_clean_resample = image.resample_img(fmri_data_clean, target_affine=target_affine)
    ##print(fmri_data_clean_resample.get_data().shape)

    ##pre-select task types
    trial_mask = pd.Series(label_trials).isin(target_name)  ##['hand', 'foot','tongue']
    fmri_data_cnn = image.index_img(fmri_data_clean_resample, np.where(trial_mask)[0]).get_data()
    ###use each slice along z-axis as one sample
    label_data_trial = np.array(label_trials.loc[trial_mask])
    img_rows, img_cols, img_deps = fmri_data_cnn.shape[:-1]

    ##cut the trials into blocks
    chunks = int(np.floor(len(label_data_trial)/block_dura))
    fmri_data_cnn_block = np.array(np.array_split(fmri_data_cnn,chunks,axis=3))
    if block_dura != fmri_data_cnn_block.shape[-1]:
        print('adjust the duration of each block from %d to %d' % (block_dura,fmri_data_cnn_block.shape[-1]))
        block_dura = fmri_data_cnn_block.shape[-1]

    label_data_trial_block = np.array_split(label_data_trial,chunks)
    label_data_trial_block = list(zip(*label_data_trial_block))[0]
    le = preprocessing.LabelEncoder()
    le.fit(target_name)
    label_data_cnn_block = le.transform(label_data_trial_block) ##np_utils.to_categorical(): convert label vector to matrix

    ###reshape data to fit the model
    fmri_data_cnn_test = fmri_data_cnn_block    ##np.transpose(fmri_data_cnn, (3, 0, 1, 2))
    label_data_cnn_test = label_data_cnn_block.flatten()
    ##print(fmri_file, fmri_data_cnn_test.shape, label_data_cnn_test.shape)

    return fmri_data_cnn_test, label_data_cnn_test

def data_pipe_3dcnn_block(fmri_files, confound_files, label_matrix, target_name=None, flag_cnn='3d', block_dura=1,
                    batch_size=32,data_type='train',train_percent=0.8, nr_thread=nr_thread, buffer_size=buffer_size):
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
    ##Subject_Num, Trial_Num = np.array(label_matrix).shape

    ####running the model
    start_time = time.clock()
    if flag_cnn == '2d':
        ds1 = dataflow.MultiThreadMapData(
            ds0, nr_thread=nr_thread,
            map_func=lambda dp: map_load_fmri_image_block(dp, target_name,block_dura=block_dura),
            buffer_size=buffer_size,
            strict=True)
    elif flag_cnn == '3d':
        ds1 = dataflow.MultiThreadMapData(
            ds0, nr_thread=nr_thread,
            map_func=lambda dp: map_load_fmri_image_3d_block(dp, target_name,block_dura=block_dura),
            buffer_size=buffer_size,
            strict=True)

    ds1 = dataflow.PrefetchData(ds1, buffer_size, 1)

    ds1 = split_samples(ds1)
    print('prefetch dataflowSize is ' + str(ds1.size()))
    ds_data_shape= ds1.data_info()
    Trial_Num = ds_data_shape[0]
    Block_dura = ds_data_shape[-1]
    print('%d #Trials/Samples per subject with %d channels in tc' % (Trial_Num,Block_dura))

    #ds1 = dataflow.LocallyShuffleData(ds1, buffer_size=ds1.size() * buffer_size)
    ds1 = dataflow.LocallyShuffleData(ds1, buffer_size=Trial_Num * buffer_size)

    ds1 = dataflow.BatchData(ds1, batch_size=batch_size)
    print('Time Usage of loading data in seconds: {} \n'.format(time.clock() - start_time))

    ds1 = dataflow.PrefetchDataZMQ(ds1, nr_proc=1)
    ds1._reset_once()
    ##ds1.reset_state()

    for df in ds1.get_data():
        if flag_cnn == '2d':
            yield (df[0].astype('float32'),to_categorical(df[1].astype('int32'), len(target_name)))
        elif flag_cnn == '3d':
            yield (df[0].astype('float32'),to_categorical(df[1].astype('int32'), len(target_name)))

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

def build_fc_nn_model(Nfeatures,Nlabels,layers=4,hidden_size=256,dropout=0.25):
    ######fully-connected neural networks
    input0 = Input(shape=(Nfeatures,))
    drop1 = input0
    for li in np.arange(layers):
        hidden1 = Dense(hidden_size, kernel_regularizer=regularizers.l2(l2_reg), activation='relu')(drop1)
        drop1 = Dropout(dropout)(hidden1)
        hidden_size = np.int32(hidden_size / 2)
        if hidden_size < 10:
            hidden_size = 16

    hidden2 = Dense(32, kernel_regularizer=regularizers.l2(l2_reg), activation='relu')(drop1)
    drop2 = Dropout(0.5)(hidden2)
    out = Dense(Nlabels, activation='softmax')(drop2)

    model = Model(inputs=input0, outputs=out)
    adam = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()

    return model

def build_cnn_model(input_shape, Nlabels, filters=32, convsize=3, poolsize=2, hidden_size=128, conv_layers=4):
    #     import keras.backend as K
    #     if K.image_data_format() == 'channels_first':
    #         img_shape = (1,img_rows,img_cols)
    #     elif K.image_data_format() == 'channels_last':
    #         img_shape = (img_rows,img_cols,1)

    input0 = Input(shape=input_shape)
    drop1 = input0
    for li in range(conv_layers):
        conv1 = Conv2D(filters, (convsize, convsize), padding='same',activation='relu',
                    kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(l2_reg))(drop1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(filters, (convsize, convsize), padding='same',activation='relu',
                    kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(l2_reg))(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D((poolsize, poolsize))(conv1)
        drop1 = Dropout(0.25)(pool1)
        if (li+1) % 2 == 0:
            filters *= 2

    drop2 = drop1
    avg1 = AveragePooling2D(pool_size=(5, 5))(drop2)
    flat = Flatten()(avg1)
    hidden = Dense(hidden_size, kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(l2_reg), activation='relu')(flat)
    drop3 = Dropout(0.4)(hidden)
    # hidden = Dense((hidden_size/4).astype(int), activation='relu')(drop3)
    # drop4 = Dropout(0.5)(hidden)

    out = Dense(Nlabels, activation='softmax')(drop3)

    model = Model(inputs=input0, outputs=out)
    adam = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()

    return model


def build_cnn3d_model(input_shape, Nlabels, filters=16, convsize=3, poolsize=2, hidden_size=128, conv_layers=4):
    #     import keras.backend as K
    #     if K.image_data_format() == 'channels_first':
    #         img_shape = (1,img_rows,img_cols,img_deps)
    #     elif K.image_data_format() == 'channels_last':
    #         img_shape = (img_rows,img_cols, img_deps,1)

    input0 = Input(shape=input_shape)
    drop1 = input0
    for li in range(conv_layers):
        conv1 = Conv3D(filters, (convsize, convsize, convsize), padding='same',activation='relu',
                    kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(l2_reg))(drop1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv3D(filters, (convsize, convsize, convsize), padding='same',activation='relu',
                    kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(l2_reg))(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling3D((poolsize, poolsize, poolsize))(conv1)
        drop1 = Dropout(0.25)(pool1)
        if (li+1) % 2 == 0:
            filters *= 2

    drop2 = drop1
    avg1 = AveragePooling3D(pool_size=(3, 3, 3))(drop2)
    flat = Flatten()(avg1)
    hidden = Dense(hidden_size, kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(l2_reg), activation='relu')(flat)
    drop3 = Dropout(0.4)(hidden)

    out = Dense(Nlabels, activation='softmax')(drop3)

    model = Model(inputs=input0, outputs=out)
    ###change optimizer to SGD with changing learning rate
    adam = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()

    return model

####change: reduce memory but increase parameters to train
def build_cnn_model_test(input_shape, Nlabels, filters=16, convsize=3, poolsize=2, hidden_size=128, conv_layers=4):
    #     import keras.backend as K
    #     if K.image_data_format() == 'channels_first':
    #         img_shape = (1,img_rows,img_cols)
    #     elif K.image_data_format() == 'channels_last':
    #         img_shape = (img_rows,img_cols,1)

    input0 = Input(shape=input_shape)
    drop1 = input0
    ####quickly reducing image dimension first
    for li in range(1):
        conv1 = Conv2D(filters, (convsize, convsize), padding='same',activation='relu',
                    kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(l2_reg))(drop1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D((poolsize, poolsize))(conv1)
        drop1 = Dropout(0.25)(pool1)
        filters *= 2
    for li in range(conv_layers-1):
        conv1 = Conv2D(filters, (convsize, convsize), padding='same',activation='relu',
                    kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(l2_reg))(drop1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(filters, (convsize, convsize), padding='same',activation='relu',
                    kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(l2_reg))(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(filters, (convsize, convsize), padding='same',activation='relu',
                    kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(l2_reg))(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D((poolsize, poolsize))(conv1)
        drop1 = Dropout(0.25)(pool1)
        if (li+1) % 2 == 0:
            filters *= 2

    drop2 = drop1
    avg1 = AveragePooling2D(pool_size=(5, 5))(drop2)
    flat = Flatten()(avg1)
    hidden = Dense(hidden_size, kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(l2_reg), activation='relu')(flat)
    drop3 = Dropout(0.4)(hidden)
    # hidden = Dense((hidden_size/4).astype(int), activation='relu')(drop3)
    # drop4 = Dropout(0.5)(hidden)

    out = Dense(Nlabels, activation='softmax')(drop3)

    model = Model(inputs=input0, outputs=out)
    adam = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()

    return model

def build_cnn3d_model_test(input_shape, Nlabels, filters=16, convsize=3, poolsize=2, hidden_size=128, conv_layers=4):
    #     import keras.backend as K
    #     if K.image_data_format() == 'channels_first':
    #         img_shape = (1,img_rows,img_cols,img_deps)
    #     elif K.image_data_format() == 'channels_last':
    #         img_shape = (img_rows,img_cols, img_deps,1)

    input0 = Input(shape=input_shape)
    drop1 = input0
    ####quickly reducing image dimension first
    for li in range(1):
        conv1 = Conv3D(filters, (convsize, convsize, convsize), padding='same',activation='relu',
                    kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(l2_reg))(drop1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling3D((poolsize, poolsize, poolsize))(conv1)
        drop1 = Dropout(0.25)(pool1)
        filters *= 2
    for li in range(conv_layers-1):
        conv1 = Conv3D(filters, (convsize, convsize, convsize), padding='same',activation='relu',
                    kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(l2_reg))(drop1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv3D(filters, (convsize, convsize, convsize), padding='same',activation='relu',
                    kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(l2_reg))(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv3D(filters, (convsize, convsize, convsize), padding='same',activation='relu',
                    kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(l2_reg))(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling3D((poolsize, poolsize, poolsize))(conv1)
        drop1 = Dropout(0.25)(pool1)
        if (li+1) % 2 == 0:
            filters *= 2

    drop2 = drop1
    avg1 = AveragePooling3D(pool_size=(3, 3, 3))(drop2)
    flat = Flatten()(avg1)
    hidden = Dense(hidden_size, kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(l2_reg), activation='relu')(flat)
    drop3 = Dropout(0.4)(hidden)

    out = Dense(Nlabels, activation='softmax')(drop3)

    model = Model(inputs=input0, outputs=out)
    ###change optimizer to SGD with changing learning rate
    adam = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()

    return model


#####################
#####
if __name__ == '__main__':

    fmri_files, confound_files, subjects = load_fmri_data(pathdata,modality)
    print('including %d fmri files and %d confounds files \n\n' % (len(fmri_files), len(confound_files)))

    label_matrix, sub_name, Block_dura = load_event_files(fmri_files,confound_files)
    if Flag_Block_Trial == 0:
        block_dura = 1
        print('\n use each trial as one sample during model training ')
    print('Collecting event design files for subjects and saved into matrix ...' , np.array(label_matrix).shape)

    nb_class = len(target_name)
    img_shape = []
    if Flag_CNN_Model == '2d':
        tc_matrix = nib.load(fmri_files[0])
        img_rows, img_cols, img_deps = tc_matrix.shape[:-1]
        if K.image_data_format() == 'channels_first':
            img_shape = (block_dura, img_rows, img_cols)
        elif K.image_data_format() == 'channels_last':
            img_shape = (img_rows, img_cols, block_dura)
    elif Flag_CNN_Model == '3d':
        fmri_data_resample = image.resample_img(fmri_files[0], target_affine=np.diag((img_resize, img_resize, img_resize)))
        img_rows, img_cols, img_deps = fmri_data_resample.get_data().shape[:-1]
        if K.image_data_format() == 'channels_first':
            img_shape = (block_dura, img_rows, img_cols, img_deps)
        elif K.image_data_format() == 'channels_last':
            img_shape = (img_rows, img_cols, img_deps, block_dura)
    print("fmri data in shape: ",img_shape)
    #########################################
    '''
    ##test whether dataflow from tensorpack works
    test_sub_num = 1000
    tst = data_pipe_3dcnn(fmri_files[:test_sub_num], confound_files[:test_sub_num], label_matrix.iloc[:test_sub_num],
                          target_name=target_name, flag_cnn=Flag_CNN_Model, batch_size=16, data_type='train', buffer_size=5)
    out = next(tst)
    print(out[0].shape)
    print(out[1].shape)

    ####################
    #####start 2dcnn model
    test_sub_num = int(len(fmri_files) * (1-test_size))
    ##xx = data_pipe(fmri_files,confound_files,label_matrix,target_name=target_name)
    train_gen = data_pipe(fmri_files[:test_sub_num],confound_files[:test_sub_num],label_matrix.iloc[:test_sub_num],
                          target_name=target_name,batch_size=32,data_type='train',nr_thread=4, buffer_size=20)
    val_set = data_pipe(fmri_files[test_sub_num:],confound_files[test_sub_num:],label_matrix.iloc[test_sub_num:],
                        target_name=target_name,batch_size=32,data_type='test',nr_thread=2, buffer_size=20)
    '''

    #########################################

    ###change:  k-fold cross-validation: split data into train, validation,test
    test_sub_num = int(len(fmri_files) * (1-test_size))
    print('\nTraining the model on %d subjects and testing on %d \n' % (test_sub_num,len(fmri_files[test_sub_num:])))
    ######start cnn model
    train_gen = data_pipe_3dcnn(fmri_files[:test_sub_num], confound_files[:test_sub_num],label_matrix.iloc[:test_sub_num],
                                target_name=target_name, flag_cnn=Flag_CNN_Model, train_percent=0.8,
                                batch_size=16, data_type='train', nr_thread=nr_thread, buffer_size=buffer_size)
    val_set = data_pipe_3dcnn(fmri_files[test_sub_num:], confound_files[test_sub_num:],label_matrix.iloc[test_sub_num:],
                              target_name=target_name, flag_cnn=Flag_CNN_Model, train_percent=0.8,
                              batch_size=16, data_type='train', nr_thread=nr_thread, buffer_size=buffer_size)
    '''
    #########################################
    test_sub_num = int(len(fmri_files) * (1-test_size))
    train_gen = data_pipe_3dcnn_block(fmri_files[:test_sub_num], confound_files[:test_sub_num],label_matrix.iloc[:test_sub_num],
                                target_name=target_name, flag_cnn=Flag_CNN_Model, train_percent=0.8, block_dura=block_dura,
                                batch_size=16, data_type='train', nr_thread=nr_thread, buffer_size=buffer_size)
    val_set = data_pipe_3dcnn_block(fmri_files[test_sub_num:], confound_files[test_sub_num:],label_matrix.iloc[test_sub_num:],
                              target_name=target_name, flag_cnn=Flag_CNN_Model, train_percent=0.8, block_dura=block_dura,
                              batch_size=16, data_type='train', nr_thread=nr_thread, buffer_size=buffer_size)
    '''
    if Flag_CNN_Model == '2d':
        print('\nTraining the model using 2d-CNN \n')
        model_test = build_cnn_model_test(img_shape, nb_class)
    elif Flag_CNN_Model == '3d':
        print('\nTraining the model using 3d-CNN \n')
        model_test = build_cnn3d_model_test(img_shape, nb_class)

    ######start training the model
    ##model_name = 'checkpoints/' + 'model_test_' + Flag_CNN_Model + 'cnn_' + modality
    model_name = "checkpoints/{}_model_test_{}cnn_lr{}_{}".format(modality,Flag_CNN_Model,learning_rate,datetime.datetime.now().strftime("%m-%d-%H"))

    tensorboard = TensorBoard(log_dir="logs/{}_{}cnn_win{}_lr{}".format(modality,Flag_CNN_Model,block_dura,learning_rate))
    early_stopping_callback = EarlyStopping(monitor='val_acc', patience=20, mode='max')
    checkpoint_callback = ModelCheckpoint(model_name+'.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    model_test_history2 = model_test.fit_generator(train_gen, epochs=100, steps_per_epoch=1000,
                                                   validation_data=val_set,validation_steps=1000, verbose=1,
                                                   callbacks=[early_stopping_callback, checkpoint_callback,tensorboard],
                                                   workers=1, use_multiprocessing=False, shuffle=True)
    print(model_test_history2.history)
    ## visualized with TensorBoad launched at the command line:
    ## tensorboard --logdir=logs/

    model_test_history = model_test.fit_generator(train_gen, epochs=50, steps_per_epoch=100, verbose=1, shuffle=True)

    #print(model_test_history.history)
    for key,val in model_test_history.history.items():
        print(' ')
        print(key, val)
        print(' ')

    scores = model_test.evaluate_generator(val_set, steps=100, workers=1)
    print(model_test.metrics_names)
    print(scores)

    import pickle
    logfilename = pathout+'train_val_scores_' + Flag_CNN_Model + 'cnn_dump2.txt'
    if os.path.isfile(logfilename):
        logfilename = logfilename.split('.')[0] + '2.txt'
    print("Saving model results to logfile: %s" % logfilename)
    file = open(logfilename, 'wb')
    pickle.dump(model_test_history2.history, file)
    pickle.dump(model_test_history.history, file)
    pickle.dump(scores, file)
    file.close()

    sys.exit(0)

'''
def generator_from_lmdb(lmdb_filename,label_matrix, target_name=None, batchSize=32, trailNum=284, is_train=True):
    if os.path.isfile(lmdb_filename):
        print("Could not find the lmdb file for task-fmri data!")
        print("Check the path: ", lmdb_filename)
    if target_name is None:
        target_name = np.unique(label_matrix)

    Subject_Num, Trial_Num = np.array(label_matrix).shape
    lmdb_env = lmdb.open(lmdb_filename, subdir=False)
    lmdb_txn = lmdb_env.begin()
    cursor = lmdb_txn.cursor()

    while True:
        np.random.shuffle(indices)
        ind = 0
        for key, value in cursor:
            batch_indices = indices[ind:ind+batchSize]
            batch_indices.sort()
            ind += batchSize

            trial_label = label_matrix.iloc[batch_indices]
            #trial_label = ev_db["trials"][batch_indices,:]
            #trial_subname = ev_db["subject"][batch_indices,:]
            fmri_data = loads(lmdb_txn.get(key))
            pathsub = Path(os.path.dirname(key.decode("utf-8")))
            fmri_subname = pathsub.parts[-3]
            if trial_subname != fmri_subname:
                print("Mis-matching subjects lists between fmri data and event design")
                continue
            if data is None or data.shape[0] != Trial_Num:
                print('fmri data shape mis-matching between subjects...')
                continue

            yield (np.expand_dims(fmri_data.astype('float32'),axis=3),to_categorical(trial_label.astype('int32'),len(target_name)))

TrailNum = 284
Nfeatures = 360
pathtemp = pathout + "temp_res/"
##ev_filename = pathtemp + modality + '_event_labels_1200R_test_ALL.h5'
lmdb_filename = pathtemp + modality + '_MMP_ROI_act_1200R_test_ALL.lmdb'
train_gen,test_gen = generator_from_lmdb(lmdb_filename,label_matrix, target_name=target_name,batchSize=32,trailNum=TrailNum,is_train=True)

model_test_fc = build_fc_nn_model(Nfeatures,nb_class)
model_test_fc_history = model_test_fc.fit_generator(train_gen, epochs=50, steps_per_epoch=100,
                                                    validation_data=test_gen,validation_steps=10,verbose=1,
                                                    workers=1, use_multiprocessing=False, shuffle=False)
print(model_test_fc_history.history)

'''
'''
from tensorpack import *
from tensorpack.tfutils import summary
from tensorpack.dataflow import dataset

class Model(ModelDesc):
    def inputs(self,img_shape):
        return [tf.placeholder(tf.float32, (None, img_shape), 'input'),
                tf.placeholder(tf.int32, (None,), 'label')]

    def build_graph(self, img_shape, label):
        image = tf.expand_dims(image, 3) * 2 - 1

        if Flag_CNN_Model == '2d':
            print('\nTraining the model using 2d-CNN \n')
            model_test = build_cnn_model(img_shape, nb_class)
        elif Flag_CNN_Model == '3d':
            print('\nTraining the model using 3d-CNN \n')
            model_test = build_cnn3d_model(img_shape, nb_class)

        logits = model_test(image)
        # build cost function by tensorflow
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')  # the average cross-entropy loss

        # for tensorpack validation
        acc = tf.to_float(tf.nn.in_top_k(logits, label, 1))
        acc = tf.reduce_mean(acc, name='accuracy')
        summary.add_moving_summary(acc)

        wd_cost = tf.add_n(model_test.losses, name='regularize_loss')    # this is how Keras manage regularizers
        cost = tf.add_n([wd_cost, cost], name='total_cost')
        summary.add_moving_summary(cost, wd_cost)
        return cost

    def optimizer(self):
        lr = tf.train.exponential_decay(
            learning_rate=1e-3,
            global_step=get_global_step_var(),
            decay_steps=468 * 10,
            decay_rate=0.3, staircase=True, name='learning_rate')
        tf.summary.scalar('lr', lr)
        return tf.train.AdamOptimizer(lr)


def get_data():
    train = BatchData(dataset.Mnist('train'), 128)
    test = BatchData(dataset.Mnist('test'), 256, remainder=True)
    return train, test

##main function
    logger.auto_set_dir()
    dataset_train, dataset_test = get_data()

    cfg = TrainConfig(
        model=Model(),
        dataflow=dataset_train,
        callbacks=[KerasPhaseCallback(True),   # for Keras training
                   ModelSaver(),
                   InferenceRunner(dataset_test,ScalarStats(['cross_entropy_loss', 'accuracy'])),],
        max_epoch=100,)

    launch_train_with_config(cfg, QueueInputTrainer())
'''
