#!/home/yuzhang/tensorflow-py3.6/bin/python3.6

# Author: Yu Zhang
# License: simplified BSD
# coding: utf-8

## ps | grep python; pkill python


from pathlib import Path
import glob
import itertools
import lmdb
import h5py
import os
import sys
import time
import datetime

import math
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import sparse
import matplotlib.pyplot as plt
###%matplotlib inline

from nilearn import signal,image,masking
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, train_test_split,ShuffleSplit
from keras.utils import np_utils

import lmdb
import tensorflow as tf
from tensorpack import dataflow
from tensorpack.utils.serialize import dumps, loads
from keras import backend as K
from keras.utils.training_utils import multi_gpu_model

try:
    # import cnn_graph
    from cnn_graph.lib import models, graph, coarsening, utils
except ImportError:
    print('Could not find the package of graph-cnn ...')
    print('Please check the location where cnn_graph is !\n')

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
    num_CPU = num_cores
    num_GPU = 0
else:
    num_CPU = 4
    num_GPU = num_cores

###check the actual No of GPUs for usage
from tensorflow.python.client import device_lib
used_GPU_avail = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
num_GPU = len(used_GPU_avail)
print('\nAvaliable GPUs for usage: %s \n' % used_GPU_avail)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

config_TF = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = tf.Session(config=config_TF)
K.set_session(session)

#########################################################
modality = 'WM' #'MOTOR'
###dict for different types of movement
motor_task_con = {"rf": "foot_mot",
                  "lf": "foot_mot",
                  "rh": "hand_mot",
                  "lh": "hand_mot",
                  "t": "tongue_mot"}
wm_task_con   =  {"2bk_body":   "body2b_wm",
                  "2bk_faces":  "face2b_wm",
                  "2bk_places": "place2b_wm",
                  "2bk_tools":  "tool2b_wm",
                  "0bk_body":   "body0b_wm",
                  "0bk_faces":  "face0b_wm",
                  "0bk_places": "place0b_wm",
                  "0bk_tools":  "tool0b_wm"}

task_contrasts = wm_task_con #motor_task_con
target_name = np.unique(pd.Series(list(task_contrasts.values())))
#target_name = np.unique(pd.Series(list(task_contrasts.values())).str.split('_',expand = True)[0])
print(target_name)


TR = 0.72
test_size = 0.2
val_size = 0.2
Flag_Block_Trial = 0 #0
block_dura = 1  #18 for motor, 39 for wm
##Trial_dura = 39
coarsening_levels = 6

AtlasName = 'MMP'
adj_mat_type = 'surface'

pathdata = '/home/yuzhang/scratch/HCP/'
pathout = pathdata + "temp_res_new/"
mmp_atlas = pathdata + "codes/HCP_S1200_GroupAvg_v1/"+"Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii"
lmdb_filename = pathout+modality+"_MMP_ROI_act_1200R_test_Dec2018_ALL.lmdb"
adj_mat_file = pathdata + 'codes/MMP_adjacency_mat_white.pconn.nii'
'''
pathdata = '/home/yu/PycharmProjects/HCP_data/'
pathout = pathdata + "temp_res_new/"
mmp_atlas = pathdata + "HCP_S1200_GroupAvg_v1/"+"Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii"
lmdb_filename = pathout+modality+"_MMP_ROI_act_1200R_test_Dec2018_ALL.lmdb"
adj_mat_file = pathout + 'MMP_adjacency_mat_white.pconn.nii'
'''

import shutil
checkpoint_dir = "cnn_graph/checkpoints/"+modality+"/win"+str(block_dura)+"/"
shutil.rmtree(checkpoint_dir,ignore_errors=True)
os.makedirs(checkpoint_dir, exist_ok=True)

##############################################
#####start collecting data for classification algorithm
def load_fmri_data(pathdata,modality=None,confound_name=None):
    ###fMRI decoding: using event signals instead of activation pattern from glm
    ##collect task-fMRI signals

    if not modality:
        modality = 'MOTOR'  # 'MOTOR'

    pathfmri = pathdata + 'aws_s3_HCP1200/FMRI/'
    pathdata = Path(pathfmri)
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


def load_event_files(pathdata,fmri_files,confound_files,ev_filename=None):
    ###collect the event design files
    tc_matrix = nib.load(fmri_files[0])
    Subject_Num = len(fmri_files)
    Trial_Num = tc_matrix.shape[-1]
    print("Data samples including %d subjects with %d trials" % (Subject_Num, Trial_Num))

    pathfmri = pathdata + 'aws_s3_HCP1200/FMRI/'
    pathdata = Path(pathfmri)
    EVS_files = []
    subj = 0
    '''
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
    '''
    ###adjust the code after changing to the new folder
    for ev, sub_count in zip(sorted(pathdata.glob('tfMRI_' + modality + '_??/*combined_events_spm_' + modality + '.csv')),range(Subject_Num)):
        ###remove fmri files if the event design is missing
        while os.path.basename(fmri_files[subj]).split('_')[0] < os.path.basename(str(ev)).split('_')[0]:
            print("Event files and fmri data are miss-matching for subject: ")
            print(os.path.basename(str(ev)).split('_')[0], ':', os.path.basename(fmri_files[subj]).split('_')[0])
            print("Due to missing event files for subject : %s" % os.path.basename(fmri_files[subj]))
            fmri_files[subj] = []
            confound_files[subj] = []
            subj += 1
            if subj > Subject_Num:
                break
        if os.path.basename(fmri_files[subj]).split('_')[0] == os.path.basename(str(ev)).split('_')[0]:
            EVS_files.append(str(ev))
            subj += 1

    fmri_files = list(filter(None, fmri_files))
    confound_files = list(filter(None, confound_files))
    if len(EVS_files) != len(fmri_files):
        print('Miss-matching number of subjects between event:{} and fmri:{} files'.format(len(EVS_files), len(fmri_files)))

    ################################
    ###loading all event designs
    if not ev_filename:
        ev_filename = "_event_labels_1200R_LR_RL_new.txt"

    events_all_subjects_file = pathout+modality+ev_filename
    if os.path.isfile(events_all_subjects_file):
        trial_infos = pd.read_csv(EVS_files[0],sep="\t",encoding="utf8",header = None,names=['onset','duration','rep','task'])
        Duras = np.ceil((trial_infos.duration/TR)).astype(int) #(trial_infos.duration/TR).astype(int)

        print('Collecting trial info from file:', events_all_subjects_file)
        subjects_trial_labels = pd.read_csv(events_all_subjects_file,sep="\t",encoding="utf8")
        ###print(subjects_trial_labels.keys())

        subjects_trial_label_matrix = subjects_trial_labels.loc[:, 'trial1':'trial' + str(Trial_Num)].values.tolist() ##convert from dataframe to list
        #subjects_trial_label_matrix = subjects_trial_labels.values.tolist()
        trialID = subjects_trial_labels['trialID']
        sub_name = subjects_trial_labels['subject'].tolist()
        coding_direct = subjects_trial_labels['coding']
        print(np.array(subjects_trial_label_matrix).shape,len(sub_name),len(np.unique(sub_name)),len(coding_direct))
    else:
        print('Loading trial info for each task-fmri file and save to csv file:', events_all_subjects_file)
        subjects_trial_label_matrix = []
        sub_name = []
        coding_direct = []
        for subj in np.arange(Subject_Num):
            pathsub = Path(os.path.dirname(EVS_files[subj]))
            #sub_name.append(pathsub.parts[-3])
            ###adjust the code after changing to the new folder
            sub_name.append(str(os.path.basename(EVS_files[subj]).split('_')[0]))
            coding_direct.append(pathsub.parts[-1].split('_')[-1])

            ##trial info in volume
            trial_infos = pd.read_csv(EVS_files[subj],sep="\t",encoding="utf8",header = None,names=['onset','duration','rep','task'])
            Onsets = np.ceil((trial_infos.onset/TR)).astype(int) #(trial_infos.onset/TR).astype(int)
            Duras = np.ceil((trial_infos.duration/TR)).astype(int) #(trial_infos.duration/TR).astype(int)
            Movetypes = trial_infos.task

            labels = ["rest"]*Trial_Num;
            trialID = [0] * Trial_Num;
            tid = 1
            for start,dur,move in zip(Onsets,Duras,Movetypes):
                for ti in range(start-1,start+dur):
                    labels[ti]= task_contrasts[move]
                    trialID[ti] = tid
                tid += 1
            subjects_trial_label_matrix.append(labels)

        ##subjects_trial_label_matrix = np.array(subjects_trial_label_matrix)
        print(np.array(subjects_trial_label_matrix).shape)
        #print(np.array(subjects_trial_label_matrix[0]))
        subjects_trial_labels = pd.DataFrame(data=np.array(subjects_trial_label_matrix),columns=['trial'+str(i+1) for i in range(Trial_Num)])
        subjects_trial_labels['trialID'] = tid
        subjects_trial_labels['subject'] = sub_name
        subjects_trial_labels['coding'] = coding_direct
        subjects_trial_labels.keys()
        #print(subjects_trial_labels['subject'],subjects_trial_labels['coding'])

        ##save the labels
        subjects_trial_labels.to_csv(events_all_subjects_file,sep='\t', encoding='utf-8',index=False)

    block_dura = np.unique(Duras)[0]
    return subjects_trial_label_matrix, sub_name, block_dura

def load_fmri_data_from_lmdb(lmdb_filename):
    ##lmdb_filename = pathout + modality + "_MMP_ROI_act_1200R_test_Dec2018_ALL.lmdb"
    ## read lmdb matrix
    print('loading data from file: %s' % lmdb_filename)
    matrix_dict = []
    fmri_sub_name = []
    lmdb_env = lmdb.open(lmdb_filename, subdir=False)
    try:
        lmdb_txn = lmdb_env.begin()
        listed_fmri_files = loads(lmdb_txn.get(b'__keys__'))
        listed_fmri_files = [l.decode("utf-8") for l in listed_fmri_files]
        print('Stored fmri data from files:')
        print(len(listed_fmri_files))
    except:
        print('Search each key for every fmri file...')

    with lmdb_env.begin() as lmdb_txn:
        cursor = lmdb_txn.cursor()
        for key, value in cursor:
            # print(key)
            if key == b'__keys__':
                continue
            pathsub = Path(os.path.dirname(key.decode("utf-8")))
            if any('REST' in string for string in lmdb_filename.split('_')):
                fmri_sub_name.append(
                    pathsub.parts[-3] + '_' + pathsub.parts[-1].split('_')[-2][-1] + '_' + pathsub.parts[-1].split('_')[
                        -1])
            else:
                # fmri_sub_name.append(pathsub.parts[-3] + '_' + pathsub.parts[-1].split('_')[-1])
                subname_info = os.path.basename(key.decode("utf-8")).split('_')
                fmri_sub_name.append('_'.join((subname_info[0], subname_info[2], subname_info[3])))
            data = loads(lmdb_txn.get(key)).astype('float32', casting='same_kind')
            if any('REST' in string for string in lmdb_filename.split('_')):
                if data is None or data.shape[0] != Trial_Num:
                    print('fmri data shape mis-matching between subjects...')
                    print('Check subject:  %s with only %d Trials \n' % (fmri_sub_name[-1], data.shape[0]))
                    del fmri_sub_name[-1]
                else:
                    matrix_dict.append(np.array(data))
            else:
                matrix_dict.append(np.array(data))
    lmdb_env.close()

    return matrix_dict, fmri_sub_name


def preclean_data_for_shape_match(subjects_tc_matrix,subjects_trial_label_matrix, fmri_sub_name, ev_sub_name):
    print("Pre-clean the fmri and event data to make sure the matching shapes between two arrays!")
    Subject_Num = np.array(subjects_tc_matrix).shape[0]
    Trial_Num, Region_Num = subjects_tc_matrix[0].shape

    if len(fmri_sub_name) != len(ev_sub_name):
        print('Warning: Mis-matching subjects list between fmri-data-matrix and trial-label-matrix')
        print(np.array(subjects_tc_matrix).shape, np.array(subjects_trial_label_matrix).shape)
        subj = 0
        if len(fmri_sub_name) > len(ev_sub_name):
            for ev, subcount in zip(ev_sub_name, range(Subject_Num)):
                ###remove fmri files if the event design is missing
                while fmri_sub_name[subj].split('_')[0] < str(ev):
                    print("Event files and fmri data are miss-matching for subject: ")
                    print(ev, ':', fmri_sub_name[subj].split('_')[0])
                    print("Due to missing event files for subject : %s" % fmri_sub_name[subj])
                    del fmri_sub_name[subj]
                    del subjects_tc_matrix[subj]
                    subj += 1
                else:
                    if subj > Subject_Num:
                        ev_sub_name.remove(ev)
                        del subjects_trial_label_matrix[subcount]
                        subj = subcount
                    if fmri_sub_name[subj].split('_')[0] == str(ev): subj += 1
            subjects_tc_matrix[subj:] = []
            fmri_sub_name[subj:] = []

        elif len(fmri_sub_name) < len(ev_sub_name):
            for fmri_file, subcount in zip(fmri_sub_name, range(len(ev_sub_name))):
                ###remove fmri files if the event design is missing
                while str(ev_sub_name[subj]) < fmri_file.split('_')[0]:
                    print("Event files and fmri data are miss-matching for subject: ")
                    print(ev_sub_name[subj], ':', fmri_file.split('_')[0])
                    print("Due to missing fmri data for subject : %s" % str(ev_sub_name[subj]))
                    del ev_sub_name[subj]
                    del subjects_trial_label_matrix[subj]
                    subj += 1
                else:
                    if subj > len(ev_sub_name):
                        fmri_sub_name.remove(fmri_file)
                        del subjects_tc_matrix[subcount]
                        subj = subcount
                    if str(ev_sub_name[subj]) == fmri_file.split('_')[0]: subj += 1
            subjects_trial_label_matrix[subj:] = []
            ev_sub_name[subj:] = []

    for subj in range(min(len(fmri_sub_name), len(ev_sub_name))):
        try:
            tsize, rsize = np.array(subjects_tc_matrix[subj]).shape
            tsize2 = len(subjects_trial_label_matrix[subj])
        except:
            print(subj == Subject_Num - 1)
            print('The end of SubjectList...\n')
        if tsize != Trial_Num or tsize2 != Trial_Num:
            if tsize2 > Trial_Num:
                ##print('Cut event data for subject %s from %d to fit event label matrix' % (fmri_sub_name[subj],tsize2))
                subjects_trial_label_matrix[subj][Trial_Num:] = []
            else:
                print(
                    'Remove subject: %s due to different trial num: %d in the fmri data' % (fmri_sub_name[subj], tsize))
                del subjects_tc_matrix[subj]
                del subjects_trial_label_matrix[subj]

        if rsize != Region_Num:
            print('Remove subject: %s due to different region num: %d in the fmri data' % (fmri_sub_name[subj], rsize))
            del subjects_tc_matrix[subj]
            del subjects_trial_label_matrix[subj]

    print('Done matching data shapes:', np.array(subjects_tc_matrix).shape, np.array(subjects_trial_label_matrix).shape)
    return subjects_tc_matrix, subjects_trial_label_matrix


def subject_cross_validation_split_trials(tc_matrix, label_matrix,target_name, sub_num=None, block_dura=18, n_folds=10, testsize=0.2, valsize=0.1,randomseed=1234):
    ##randomseed=1234;testsize = 0.2;n_folds=10;valsize=0.1
    from sklearn import preprocessing
    from sklearn.model_selection import cross_val_score, train_test_split,ShuffleSplit
    
    Subject_Num, Trial_Num, Region_Num = np.array(tc_matrix).shape
    rs = np.random.RandomState(randomseed)
    if not sub_num or sub_num>Subject_Num:
        sub_num = Subject_Num
    if not block_dura:
        block_dura = 18 ###12s block for MOTOR task

    fmri_data_matrix = []
    label_data_matrix = []
    for subi in range(Subject_Num):
        label_trial_data = np.array(label_matrix[subi])
        condition_mask = pd.Series(label_trial_data).isin(target_name)
        ##condition_mask = pd.Series(label_trial_data).str.split('_', expand=True)[0].isin(target_name)
        fmri_data_matrix.append(tc_matrix[subi][condition_mask, :])
        label_data_matrix.append(label_trial_data[condition_mask])
    fmri_data_matrix = np.array(fmri_data_matrix).astype('float32', casting='same_kind')
    label_data_matrix = np.array(label_data_matrix)
    ##cut the trials into blocks
    chunks = int(np.floor(label_data_matrix.shape[-1] / block_dura))
    fmri_data_block = np.array(np.array_split(fmri_data_matrix, chunks, axis=1)).mean(axis=2).astype('float32',casting='same_kind')
    label_data_block = np.array(np.array_split(label_data_matrix, chunks, axis=1))[:, :, 0]
    print(fmri_data_block.shape,label_data_block.shape)

    train_sid_tmp, test_sid = train_test_split(range(sub_num), test_size=testsize, random_state=rs, shuffle=True)
    fmri_data_train = np.array([fmri_data_block[:, i, :] for i in train_sid_tmp]).astype('float32', casting='same_kind')
    fmri_data_test = np.array([fmri_data_block[:, i, :] for i in test_sid]).astype('float32', casting='same_kind')
    # print(fmri_data_train.shape,fmri_data_test.shape)

    label_data_train = np.array([label_data_block[:, i] for i in train_sid_tmp])
    label_data_test = np.array([label_data_block[:, i] for i in test_sid])
    # print(label_data_train.shape,label_data_test.shape)

    ###transform the data
    scaler = preprocessing.StandardScaler().fit(np.vstack(fmri_data_train))
    ##fmri_data_train = scaler.transform(fmri_data_train)
    X_test = scaler.transform(np.vstack(fmri_data_test))
    nb_class = len(np.unique(label_data_block))
    Y_test = label_data_test.ravel()
    # print(X_test.shape,Y_test.shape)

    from sklearn.model_selection import ShuffleSplit
    valsplit = ShuffleSplit(n_splits=n_folds, test_size=valsize, random_state=rs)
    X_train_scaled = []
    X_val_scaled = []
    Y_train_scaled = []
    Y_val_scaled = []
    for train_sid, val_sid in valsplit.split(train_sid_tmp):
        ##preprocess features and labels
        X = np.array(np.vstack([fmri_data_train[i, :, :] for i in train_sid]))
        Y = np.array([label_data_train[i, :] for i in train_sid]).ravel()
        # print(X.shape, Y.shape)
        X_train_scaled.append(scaler.transform(X))
        Y_train_scaled.append(Y)

        X = np.array(np.vstack([fmri_data_train[i, :, :] for i in val_sid]))
        Y = np.array([label_data_train[i, :] for i in val_sid]).ravel()
        # print(X.shape, Y.shape)
        X_val_scaled.append(scaler.transform(X))
        Y_val_scaled.append(Y)

    print('Samples of Subjects for training: %d and testing %d and validating %d with %d classes' % (len(train_sid), len(test_sid), len(val_sid), nb_class))
    return X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled, X_test, Y_test


def subject_cross_validation_split_trials_new(tc_matrix, label_matrix, target_name, sub_num=None, block_dura=18,
                                              n_folds=10, testsize=0.2, valsize=0.1, randomseed=1234):
    ##randomseed=1234;testsize = 0.2;n_folds=10;valsize=0.1
    from sklearn import preprocessing
    from sklearn.model_selection import cross_val_score, train_test_split, ShuffleSplit

    Subject_Num, Trial_Num, Region_Num = np.array(tc_matrix).shape
    rs = np.random.RandomState(randomseed)
    if not sub_num or sub_num > Subject_Num:
        sub_num = Subject_Num
    if not block_dura:
        block_dura = 18  ###12s block for MOTOR task

    global Trial_dura
    print('each trial contains %d volumes/TRs for task %s' % (Trial_dura, modality))

    fmri_data_matrix = []
    label_data_matrix = []
    for subi in range(Subject_Num):
        label_trial_data = np.array(label_matrix[subi])
        condition_mask = pd.Series(label_trial_data).isin(target_name)
        ##condition_mask = pd.Series(label_trial_data).str.split('_', expand=True)[0].isin(target_name)

        tc_matrix_select = np.array(tc_matrix[subi][condition_mask, :])
        label_data_select = np.array(label_trial_data[condition_mask])
        ##print(tc_matrix_select.shape,label_data_select.shape)

        le = preprocessing.LabelEncoder()
        le.fit(target_name)
        label_data_int = le.transform(label_data_select)

        ##cut the trials
        chunks = int(np.floor(len(label_data_select) / Trial_dura))
        label_data_trial_block = np.array(np.split(label_data_select, np.where(np.diff(label_data_int))[0] + 1))
        fmri_data_block = np.array_split(tc_matrix_select, np.where(np.diff(label_data_int))[0] + 1, axis=0)
        if subi == 1:
            ulabel = [np.unique(x) for x in label_data_trial_block]
            print("After cutting: unique values for each block of trials %s with %d blocks" % (np.array(ulabel), len(ulabel)))
        if label_data_trial_block.shape[0] != chunks:
            label_data_trial_block = np.array(np.split(label_data_select, chunks))
            fmri_data_block = np.array_split(tc_matrix_select, chunks, axis=0)
            if subi == 1:
                print("\nWrong cutting of event data...")
                print("Should have %d block-trials but only found %d cuts" % (chunks, label_data_trial_block.shape[0]))
                ulabel = [np.unique(x) for x in label_data_trial_block]
                print("Adjust the cutting: unique values for each block of trials %s with %d blocks\n" % (np.array(ulabel), len(ulabel)))

        label_data_trial_block = np.array([label_data_trial_block[i][:Trial_dura] for i in range(chunks)])
        fmri_data_block = np.array([fmri_data_block[i][:Trial_dura, :] for i in range(chunks)])
        if subi == 1: print('first cut:', fmri_data_block.shape, label_data_trial_block.shape)

        ##cut each trial to blocks
        chunks = int(np.ceil(Trial_dura / block_dura))
        if Trial_dura % block_dura:
            fmri_data_block = np.array(np.vstack(np.array_split(fmri_data_block, chunks, axis=1)[:-1])).mean(axis=1).astype('float32', casting='same_kind')
            label_data_trial_block = np.array(np.vstack(np.array_split(label_data_trial_block, chunks, axis=1)[:-1]))[:,0]
        else:
            fmri_data_block = np.array(np.vstack(np.array_split(fmri_data_block, chunks, axis=1))).mean(axis=1).astype('float32', casting='same_kind')
            label_data_trial_block = np.array(np.vstack(np.array_split(label_data_trial_block, chunks, axis=1)))[:, 0]
        if subi == 1: print('second cut:', fmri_data_block.shape, label_data_trial_block.shape)
        ##label_data_test = le.transform(label_data_trial_block[:,0]).flatten()
        if subi == 1: print('finalize: reshape data into size:', fmri_data_block.shape, label_data_trial_block.shape)

        fmri_data_matrix.append(fmri_data_block)
        label_data_matrix.append(label_data_trial_block)
    fmri_data_matrix = np.array(fmri_data_matrix).astype('float32', casting='same_kind')
    label_data_matrix = np.array(label_data_matrix)
    print(fmri_data_matrix.shape, label_data_matrix.shape)

    ########spliting into train,val and testing
    train_sid_tmp, test_sid = train_test_split(range(sub_num), test_size=testsize, random_state=rs, shuffle=True)
    if len(train_sid_tmp)<2 or len(test_sid)<2:
        print("Only %d subjects avaliable. Use all subjects for training and testing" % (sub_num))
        train_sid_tmp = range(sub_num)
        test_sid = range(sub_num)
    fmri_data_train = np.array([fmri_data_matrix[i] for i in train_sid_tmp]).astype('float32', casting='same_kind')
    fmri_data_test = np.array([fmri_data_matrix[i] for i in test_sid]).astype('float32', casting='same_kind')
    print('fmri data for train and test:', fmri_data_train.shape, fmri_data_test.shape)

    label_data_train = np.array([label_data_matrix[i] for i in train_sid_tmp])
    label_data_test = np.array([label_data_matrix[i] for i in test_sid])
    print('label data for train and test', label_data_train.shape, label_data_test.shape)

    ###transform the data

    scaler = preprocessing.StandardScaler().fit(np.vstack(fmri_data_train))
    ##fmri_data_train = scaler.transform(fmri_data_train)
    X_test = scaler.transform(np.vstack(fmri_data_test))
    nb_class = len(target_name)
    Y_test = label_data_test.ravel()
    ##print(X_test.shape,Y_test.shape)

    from sklearn.model_selection import ShuffleSplit
    valsplit = ShuffleSplit(n_splits=n_folds, test_size=valsize, random_state=rs)
    X_train_scaled = []
    X_val_scaled = []
    Y_train_scaled = []
    Y_val_scaled = []
    for train_sid, val_sid in valsplit.split(train_sid_tmp):
        ##preprocess features and labels
        X = np.array(np.vstack([fmri_data_train[i] for i in train_sid]))
        Y = np.array([label_data_train[i] for i in train_sid]).ravel()
        # print('fmri and label data for training:',X.shape, Y.shape)
        X_train_scaled.append(scaler.transform(X))
        Y_train_scaled.append(Y)

        X = np.array(np.vstack([fmri_data_train[i] for i in val_sid]))
        Y = np.array([label_data_train[i] for i in val_sid]).ravel()
        # print('fmri and label data for validation:',X.shape, Y.shape)
        X_val_scaled.append(scaler.transform(X))
        Y_val_scaled.append(Y)

    print('Samples of Subjects for training: %d and testing %d and validating %d with %d classes' % (
    len(train_sid), len(test_sid), len(val_sid), nb_class))
    return X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled, X_test, Y_test


def gccn_model_common_param(modality,training_samples,target_name=None,nepochs=50,batch_size=128,layers=3,pool_size=4,hidden_size=256):
    ###common settings for gcn models
    C = len(target_name)+1
    global block_dura

    gcnn_common = {}
    gcnn_common['dir_name'] = modality + '/' + 'win' + str(block_dura) + '/'
    gcnn_common['num_epochs'] = nepochs
    gcnn_common['batch_size'] = batch_size
    gcnn_common['decay_steps'] = training_samples / gcnn_common['batch_size']  ##refine this according to samples
    gcnn_common['eval_frequency'] = 30 * gcnn_common['num_epochs']
    gcnn_common['brelu'] = 'b1relu'
    gcnn_common['pool'] = 'mpool1'

    ##more params on conv
    # Common hyper-parameters for LeNet5-like networks (two convolutional layers).
    gcnn_common['regularization'] = 5e-4
    gcnn_common['dropout'] = 0.5
    gcnn_common['learning_rate'] = 0.02  # 0.03 in the paper but sgconv_sgconv_fc_softmax has difficulty to converge
    gcnn_common['decay_rate'] = 0.95
    gcnn_common['momentum'] = 0.9
    gcnn_common['F'] = [32 * math.pow(2, li) for li in
                        range(layers)]  # [32, 64, 128]  # Number of graph convolutional filters.
    gcnn_common['K'] = [25 for li in range(layers)]  # [25, 25, 25]  # Polynomial orders.
    gcnn_common['p'] = [pool_size for li in range(layers)]  # [4, 4, 4]  # Pooling sizes.
    gcnn_common['M'] = [hidden_size, C]  # Output dimensionality of fully connected layers.
    return gcnn_common

def build_graph_adj_mat(adjacent_mat_file,adjacent_mat_type,coarsening_levels=6,Nneighbours=8, noise_level=0.001):
    ##loading the first-level graph adjacent matrix based on surface neighbourhood
    if adjacent_mat_type.lower() == 'surface':
        print('\n\nLoading adjacent matrix based on counting connected vertices between parcels')

        adj_mat = nib.load(adjacent_mat_file).get_data()
        adj_mat = sparse.csr_matrix(adj_mat)
    elif adjacent_mat_type.lower() == 'sc':
        print('\n\nCalculate adjacent graph based on structural covaraince of corrThickness across subjects')
        conn_matrix = nib.load(adjacent_mat_file).get_data()

        global mmp_atlas
        atlas_roi = nib.load(mmp_atlas).get_data()
        RegionLabels = [i for i in np.unique(atlas_roi) if i != 0]

        conn_roi_matrix = []
        for li in sorted(RegionLabels):
            tmp_ind = [ind for ind in range(conn_matrix.shape[1]) if atlas_roi[0][ind] == li]
            conn_roi_matrix.append(np.mean(conn_matrix[:, tmp_ind], axis=1))
        conn_roi_matrix = np.transpose(np.array(conn_roi_matrix))
        dist, idx = graph.distance_sklearn_metrics(np.transpose(conn_roi_matrix), k=Nneighbours, metric='cosine')
        adj_mat = graph.adjacency(dist, idx)

    print(adj_mat.shape)
    A = graph.replace_random_edges(adj_mat, noise_level)

    return A

def build_fourier_graph_cnn(gcnn_common,Laplacian_list=None):
    if not Laplacian_list:
        print('Laplacian matrix for multi-scale graphs are requried!')
    else:
        print('Laplacian matrix for multi-scale graphs:')
        print([Laplacian_list[li].shape for li in range(len(Laplacian_list))])

    ##model#1: two convolutional layers with fourier transform as filters
    name = 'fgconv_fgconv_fc_softmax'  # 'Non-Param'
    params = gcnn_common.copy()
    params['dir_name'] += name
    params['filter'] = 'fourier'
    params['K'] = np.zeros(len(gcnn_common['p']), dtype=int)
    for pi, li in zip(gcnn_common['p'], range(len(gcnn_common['p']))):
        if pi == 2:
            params['K'][li] = Laplacian_list[li].shape[0]
        if pi == 4:
            params['K'][li] = Laplacian_list[li * 2].shape[0]
    print(params)

    model = models.cgcnn(config_TF, Laplacian_list, **params)

    print('\nBuilding convolutional layers with fourier basis of Laplacian\n')
    print([Laplacian_list[li].shape for li in range(len(Laplacian_list))])
    return model, name, params


def build_spline_graph_cnn(gcnn_common, Laplacian_list=None):
    if not Laplacian_list:
        print('Laplacian matrix for multi-scale graphs are requried!')
    else:
        print('Laplacian matrix for multi-scale graphs:')
        print([Laplacian_list[li].shape for li in range(len(Laplacian_list))])

    ##model#2: two convolutional layers with spline basis as filters
    name = 'sgconv_sgconv_fc_softmax'  # 'Non-Param'
    params = gcnn_common.copy()
    params['dir_name'] += name
    params['filter'] = 'spline'
    print(params)

    model = models.cgcnn(config_TF, Laplacian_list, **params)
    print('\nBuilding convolutional layers with spline basis\n')
    print([Laplacian_list[li].shape for li in range(len(Laplacian_list))])
    return model, name, params

def build_chebyshev_graph_cnn(gcnn_common, Laplacian_list=None):
    gcnn_common['learning_rate'] = 0.005  # 0.03 in the paper but sgconv_sgconv_fc_softmax has difficulty to converge
    gcnn_common['decay_rate'] = 0.9  ##0.95

    if not Laplacian_list:
        print('Laplacian matrix for multi-scale graphs are requried!')
    else:
        print('Laplacian matrix for multi-scale graphs:')
        print([Laplacian_list[li].shape for li in range(len(Laplacian_list))])

    ##model#3: two convolutional layers with Chebyshev polynomial as filters
    name = 'cgconv_cgconv_fc_softmax'  # 'Non-Param'
    params = gcnn_common.copy()
    params['dir_name'] += name
    params['filter'] = 'chebyshev5'
    print(params)

    model = models.cgcnn(config_TF,Laplacian_list, **params)
    print('\nBuilding convolutional layers with Chebyshev polynomial\n')
    print([Laplacian_list[li].shape for li in range(len(Laplacian_list))])
    return model, name, params

def show_gcn_results(s, fontsize=None):
    if fontsize:
        plt.rc('pdf', fonttype=42)
        plt.rc('ps', fonttype=42)
        plt.rc('font', size=fontsize)         # controls default text sizes
        plt.rc('axes', titlesize=fontsize)    # fontsize of the axes title
        plt.rc('axes', labelsize=fontsize)    # fontsize of the x any y labels
        plt.rc('xtick', labelsize=fontsize)   # fontsize of the tick labels
        plt.rc('ytick', labelsize=fontsize)   # fontsize of the tick labels
        plt.rc('legend', fontsize=fontsize)   # legend fontsize
        plt.rc('figure', titlesize=fontsize)  # size of the figure title
    print('  accuracy        F1             loss        time [ms]  name')
    print('test  train   test  train   test     train')
    for name in sorted(s.names):
        print('{:5.2f} {:5.2f}   {:5.2f} {:5.2f}   {:.2e} {:.2e}   {:3.0f}   {}'.format(
                s.test_accuracy[name], s.train_accuracy[name],
                s.test_f1[name], s.train_f1[name],
                s.test_loss[name], s.train_loss[name], s.fit_time[name]*1000, name))
    '''
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    for name in sorted(s.names):
        steps = np.arange(len(s.fit_accuracies[name])) + 1
        steps *= s.params[name]['eval_frequency']
        ax[0].plot(steps, s.fit_accuracies[name], '.-', label=name)
        ax[1].plot(steps, s.fit_losses[name], '.-', label=name)
    ax[0].set_xlim(min(steps), max(steps))
    ax[1].set_xlim(min(steps), max(steps))
    ax[0].set_xlabel('step')
    ax[1].set_xlabel('step')
    ax[0].set_ylabel('validation accuracy')
    ax[1].set_ylabel('training loss')
    ax[0].legend(loc='lower right')
    ax[1].legend(loc='upper right')
    #fig.savefig('training.pdf')
    '''

    return s

def build_graph_cnn_subject_validation(subjects_tc_matrix,subjects_trial_label_matrix,target_name,block_dura=1,
                                       layers=3,pool_size=4,hidden_size=256,batch_size=128,nepochs=100,
                                       flag_multi_gcn_compare=0, my_cv_fold=10,testsize=0.2,valsize=0.2):
    ###classification using graph convolution neural networks with subject-specific split of train, val and test
    Subject_Num = np.array(subjects_tc_matrix).shape[0]
    Trial_Num, Region_Num = np.array(subjects_tc_matrix[0]).shape
    if Trial_Num != np.array(subjects_trial_label_matrix).shape[1]:
        print('Miss-matching trial infos for event and fmri data')
    if Subject_Num != np.array(subjects_trial_label_matrix).shape[0]:
        print('Adjust subject numbers for event data')
        print('Need to run preclean_data before to ensure size matching between fmri and event data!')
        Subject_Num = min(np.array(subjects_tc_matrix).shape[0],np.array(subjects_trial_label_matrix).shape[0])
        subjects_trial_label_matrix = np.array(subjects_trial_label_matrix[:Subject_Num])
    else:
        subjects_trial_label_matrix = np.array(subjects_trial_label_matrix)


    ##split data into train, val and test in subject-level
    X_train, Y_train, X_val, Y_val, X_test, Y_test = \
        subject_cross_validation_split_trials_new(subjects_tc_matrix, subjects_trial_label_matrix, target_name,
                                                  block_dura=block_dura,n_folds=my_cv_fold, testsize=testsize, valsize=valsize)

    X_train_all = np.array(np.vstack((X_train[0], X_val[0])))
    Y_train_all = np.array(np.concatenate((Y_train[0], Y_val[0]), axis=0))
    print('sample size for training and testing: ', X_train_all.shape, Y_train_all.shape)

    ##################################################################
    ###prepare for gcn model
    ###pre-setting of common parameters
    global modality, adj_mat_file,adj_mat_type, coarsening_levels
    gcnn_common = gccn_model_common_param(modality,X_train[0].shape[0],target_name,
                                          layers=layers,pool_size=pool_size, hidden_size=hidden_size,batch_size=batch_size,
                                          nepochs=nepochs)

    A = build_graph_adj_mat(adj_mat_file,adj_mat_type)
    ###build multi-level graph using coarsen (div by 2 at each level)
    graphs, perm = coarsening.coarsen(A, levels=coarsening_levels, self_connections=False)
    L = [graph.laplacian(A, normalized=True) for A in graphs]
    model_perf = utils.model_perf()

    if flag_multi_gcn_compare:
        from collections import namedtuple

        Record = namedtuple("gcnn_name", ["gcnn_model", "gcnn_params"])
        ##s = {"test_id1": Record("res1", "time1"), "test_id2": Record("res2", "time2")}
        ##s["test_id1"].resultValue

        model2, gcnn_name2, params2 = build_fourier_graph_cnn(gcnn_common,Laplacian_list=L)
        model3, gcnn_name3, params3 = build_spline_graph_cnn(gcnn_common,Laplacian_list=L)
        model4, gcnn_name4, params4 = build_chebyshev_graph_cnn(gcnn_common,Laplacian_list=L)
        gcnn_model_dicts = {#gcnn_name2: Record(model2,params2),
                            #gcnn_name3: Record(model3,params3),
                            gcnn_name4: Record(model4,params4)}
        ##initalization
        train_acc = {};
        train_loss = {};
        test_acc = {};
        test_loss = {};
        val_acc = {};
        val_loss = {};
        for name in gcnn_model_dicts.keys():
            train_acc[name] = []
            train_loss[name] = []
            test_acc[name] = []
            test_loss[name] = []
            val_acc[name] = []
            val_loss[name] = []

        ###subject-specific cross-validation
        d = {k: v + 1 for v, k in enumerate(sorted(set(Y_test)))}
        test_labels = np.array([d[x] for x in Y_test])
        print(np.unique(Y_test))


        for name in gcnn_model_dicts.keys():
            print('\n\nTraining graph cnn using %s filters!' % name)
            ###training
            model = gcnn_model_dicts[name].gcnn_model
            params = gcnn_model_dicts[name].gcnn_params
            print(name, params)

            accuracy=[]; loss=[];  t_step=[];
            for x_train, y_train, x_val, y_val, tcount in zip(X_train, Y_train, X_val, Y_val, range(len(X_train))):
                train_data = coarsening.perm_data(x_train.reshape(-1, Region_Num), perm)
                train_labels = np.array([d[x] for x in y_train])
                val_data = coarsening.perm_data(x_val.reshape(-1, Region_Num), perm)
                val_labels = np.array([d[x] for x in y_val])
                test_data = coarsening.perm_data(X_test.reshape(-1, Region_Num), perm)
                print('\nFold #%d: training on %d samples with %d features, validating on %d samples and testing on %d samples' %
                    (tcount + 1, train_data.shape[0], train_data.shape[1], val_data.shape[0], test_data.shape[0]))

                #acc, los, tstep = model.fit(train_data, train_labels, val_data, val_labels)
                #accuracy.append(acc)
                #loss.append(los)
                #t_step.append(tstep)

                ##evaluation
                model_perf.test(model, name, params,
                                train_data, train_labels, val_data, val_labels, test_data, test_labels)
                train_acc[name].append(model_perf.train_accuracy[name])
                train_loss[name].append(model_perf.train_loss[name])
                test_acc[name].append(model_perf.test_accuracy[name])
                test_loss[name].append(model_perf.test_loss[name])
                val_acc[name].append(model_perf.fit_accuracies[name])
                val_loss[name].append(model_perf.fit_losses[name])

            print('\nResults for graph-cnn using %s filters!' % name)
            print('Accuracy of training:{},testing:{}'.format(np.mean(train_acc[name]), np.mean(test_acc[name])))
            print('Accuracy of validation:mean=%2f' % np.mean(np.max(val_acc[name], axis=1)))

    else:
        ##model#1: two convolutional layers with fourier transform as filters
        model, name, params = build_fourier_graph_cnn(gcnn_common, Laplacian_list=L)

        d = {k: v + 1 for v, k in enumerate(sorted(set(Y_test)))}
        test_labels = np.array([d[x] for x in Y_test])
        print(np.unique(Y_test))

        train_acc = [];
        train_loss = [];
        test_acc = [];
        test_loss = [];
        val_acc = [];
        val_loss = [];
        accuracy = [];
        loss = [];
        t_step = [];
        for x_train, y_train, x_val, y_val, tcount in zip(X_train, Y_train, X_val, Y_val, range(2)):
            train_data = coarsening.perm_data(x_train.reshape(-1, Region_Num), perm)
            train_labels = np.array([d[x] for x in y_train])
            val_data = coarsening.perm_data(x_val.reshape(-1, Region_Num), perm)
            val_labels = np.array([d[x] for x in y_val])
            test_data = coarsening.perm_data(X_test.reshape(-1, Region_Num), perm)
            print('\nFold #%d: training on %d samples with %d features, validating on %d samples and testing on %d samples'
                  % (tcount + 1, train_data.shape[0], train_data.shape[1], val_data.shape[0], test_data.shape[0]))

            ###training
            #model = models.cgcnn(config_TF, L, **params)
            acc, los, tstep = model.fit(train_data, train_labels, val_data, val_labels)
            accuracy.append(acc)
            loss.append(los)
            t_step.append(tstep)

            ##evaluation
            model_perf.test(model, name, params,
                            train_data, train_labels, val_data, val_labels, test_data, test_labels)
            train_acc.append(model_perf.train_accuracy[name])
            train_loss.append(model_perf.train_loss[name])
            test_acc.append(model_perf.test_accuracy[name])
            test_loss.append(model_perf.test_loss[name])
            val_acc.append(model_perf.fit_accuracies[name])
            val_loss.append(model_perf.fit_losses[name])
            print('\n')

    return train_acc, test_acc, val_acc, model_perf

##################################
####main function
if __name__ == '__main__':
    with tf.device("/cpu:0"):
        fmri_files, confound_files, subjects = load_fmri_data(pathdata,modality)
        print('including %d fmri files and %d confounds files \n\n' % (len(fmri_files), len(confound_files)))

        ev_filename = "_event_labels_1200R_test_Dec2018_ALL_new.h5"
        subjects_trial_label_matrix, sub_name_ev, Trial_dura = load_event_files(pathdata, fmri_files, confound_files,ev_filename)
        if Flag_Block_Trial == 0:
            block_dura = 1
            print('\n use each trial as one sample during model training ')
        print('each trial contains %d volumes/TRs for task %s' % (Trial_dura,modality))
        print('Collecting event design files for subjects and saved into matrix ...\n' , np.array(subjects_trial_label_matrix).shape)

        print('Collecting fmri data from lmdb file')
        subjects_tc_matrix, subname_coding = load_fmri_data_from_lmdb(lmdb_filename)
        print(np.array(subjects_tc_matrix).shape)
        print('\n')

        #####
        subjects_tc_matrix_new, subjects_trial_label_matrix_new = preclean_data_for_shape_match(subjects_tc_matrix,
                                                                                                subjects_trial_label_matrix,
                                                                                                subname_coding, sub_name_ev)
        Subject_Num = np.array(subjects_tc_matrix_new).shape[0]
        Region_Num = np.array(subjects_tc_matrix_new).shape[-1]
        print(np.array(subjects_trial_label_matrix_new).shape)
        print(np.array(subjects_tc_matrix_new).shape)
        print(np.unique(subjects_trial_label_matrix_new))



    ##################################################################
    ###prepare for gcn model
    train_acc, test_acc, val_acc, model_perf = \
        build_graph_cnn_subject_validation(subjects_tc_matrix, subjects_trial_label_matrix,target_name,
                                           block_dura=block_dura, ##nepochs=10,batch_size=4,my_cv_fold=2,
                                           flag_multi_gcn_compare=1)

    print(train_acc, test_acc, val_acc)
    for gcnn_name in train_acc.keys():
        print('GCN model:',gcnn_name)
        print('Accuracy of training:{},val:{}, testing:{}'.
              format(np.mean(train_acc[gcnn_name]), np.mean(np.max(val_acc[gcnn_name], axis=1)), np.mean(test_acc[gcnn_name])))
    ###summarize the results
    #model_perf.show()
    ss = show_gcn_results(model_perf)

