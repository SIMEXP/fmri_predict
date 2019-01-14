#!/home/yuzhang/jupyter_py3/bin/python

# Author: Yu Zhang
# License: simplified BSD
# coding: utf-8

###fMRI decoding: using event signals instead of activation pattern from glm

from pathlib import Path
import glob
import re
import os
import sys
import time
import argparse
import itertools

import math
import numpy as np
import pandas as pd
import h5py
import nibabel as nib
from collections import defaultdict
import matplotlib.pyplot as plt
###%matplotlib inline

from sklearn import linear_model
from sklearn import svm, metrics
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, train_test_split,ShuffleSplit
from sklearn.decomposition import PCA, FastICA, FactorAnalysis, DictionaryLearning, KernelPCA

try:
    from keras.utils import np_utils
    from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
    from keras.models import Model
except ImportError:
    print("Tensorflow is not avaliable in the current node!")
    print("deep learning models will not be running for this test!")


from tensorpack.utils import logger
from tensorpack import dataflow
from tensorpack.utils.utils import get_tqdm
from tensorpack.utils.serialize import dumps, loads
import lmdb


##regression method to regress out head movement and other confounds
regr = linear_model.LinearRegression()

pathfmri = '/home/yuzhang/projects/rrg-pbellec/DATA/HCP/aws-s3_copy_022718/'
pathout = '/home/yuzhang/projects/rrg-pbellec/yuzhang/HCP/'
TR = 0.72
window_size = 6
##window_size_trial = math.ceil(window_size/TR)

##task info
modality = 'MOTOR'
###dict for different types of movement
task_contrasts = {"rf": "foot",
                  "lf": "foot",
                  "rh": "hand",
                  "lh": "hand",
                  "t": "tongue"}

##the chosen atlas to map fmri data
# mmp_atlas = "/home/yuzhang/projects/rrg-pbellec/yuzhang/HCP/codes/HCP_S1200_GroupAvg_v1/Gordon333.32k_fs_LR.dlabel.nii"
mmp_atlas = "/home/yuzhang/projects/rrg-pbellec/yuzhang/HCP/codes/HCP_S1200_GroupAvg_v1/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii"
AtlasName = 'MMP'
Subject_Num = 1086
Trial_Num = 284
Node_Num = 32000
Region_Num = 200

startsub = 0
endsub = Subject_Num
subjectlist = ''

def bulid_dict_task_modularity(modality):
    ##build the dict for different subtypes of events under different modalities
    motor_task_con = {"rf": "foot_mot",
                      "lf": "foot_mot",
                      "rh": "hand_mot",
                      "lh": "hand_mot",
                      "t": "tongue_mot"}
    lang_task_con =  {"present_math":  "math_lang",
                      "question_math": "math_lang",
                      "response_math": "math_lang",
                      "present_story":  "story_lang",
                      "question_story": "story_lang" ,
                      "response_story": "story_lang"}
    emotion_task_con={"fear": "fear_emo",
                      "neut": "non_emo"}
    gambl_task_con = {"win_event":  "win_gamb",
                      "loss_event": "loss_gamb",
                      "neut_event": "non_gamb"}
    reson_task_con = {"match":    "match_reson",
                      "relation": "relat_reson",
                      "error":    "mis_reson"}
    social_task_con ={"mental": "mental_soc",
                      "rnd":  "random_soc"}
    wm_task_con   =  {"2bk_body":   "body2b_wm",
                      "2bk_faces":  "face2b_wm",
                      "2bk_places": "place2b_wm",
                      "2bk_tools":  "tool2b_wm",
                      "0bk_body":   "body0b_wm",
                      "0bk_faces":  "face0b_wm",
                      "0bk_places": "place0b_wm",
                      "0bk_tools":  "tool0b_wm"}

    dicts = [motor_task_con,lang_task_con,emotion_task_con,gambl_task_con,reson_task_con,social_task_con,wm_task_con]
    all_task_con = defaultdict(list)  # uses set to avoid duplicates
    for d in dicts:
        for k, v in d.items():
            all_task_con[k].append(v)  ## all_task_con[k]=v to remove the list []

    mod_chosen = modality[:3].lower().strip()
    choices = {'mot': motor_task_con,
               'lan': lang_task_con,
               'emo': emotion_task_con,
               'gam': gambl_task_con,
               'rel': reson_task_con,
               'soc': social_task_con,
               'wm': wm_task_con,
               'all': all_task_con}

    global task_contrasts
    task_contrasts = choices.get(mod_chosen,'default')
    return task_contrasts


def load_fmri_data(pathfmri,modality,postfix=None,confound_name=None):
    ###collect fmri time-series from fmri folders
    pathdata = Path(pathfmri)
    print("Collecting fmri data from folder: %s" % pathfmri)

    if not postfix:
        postfix = ".nii.gz"
    fmri_files = [];
    confound_files = [];
    for file in sorted(pathdata.glob('*/FMRI/tfMRI_' + modality + '*/tfMRI_' + modality + '*_' + postfix)):
        fmri_files.append(str(file))

    if confound_name:
        for confound in sorted(pathdata.glob('*/FMRI/tfMRI_' + modality + '*/'+confound_name)):
            confound_files.append(str(confound))

    global startsub, endsub, subjectlist
    if not startsub:
        startsub = 0
        if not endsub:
            subjectlist = ''
    if (not endsub) or (endsub > len(fmri_files)):
        endsub = len(fmri_files)
    fmri_files = fmri_files[startsub:endsub]
    confound_files = confound_files[startsub:endsub]

    global Subject_Num, Trial_Num, Node_Num
    Subject_Num = len(fmri_files)
    tc_matrix = nib.load(fmri_files[0])
    Trial_Num, Node_Num = tc_matrix.shape
    print("Data samples including %d subjects with %d trials and %d nodes for each" % (Subject_Num, Trial_Num, Node_Num))

    ##print(globals())
    return fmri_files, confound_files


def combine_events_spm(fmri_files):
    ###combine the event design files into one csv file
    ###event design files are in the same folder of fmri files
    print("Different types of %s :" % modality)
    print(task_contrasts.keys())

    #Subject_Num = len(fmri_files)
    for subj in range(Subject_Num):
        pathsub = Path(os.path.dirname(fmri_files[subj]))
        evfiles_combined = str(pathsub)+"/combined_events_spm_"+modality+".csv"
        if not os.path.isfile(evfiles_combined):
            evs = []
            ev_names = []
            if not os.path.isdir(str(pathsub) + '/EVs/'):
                print('Event files not exist!\n Check data folder:%s' % str(pathsub))
                continue
            for ev in sorted(pathsub.glob('EVs/*.txt')):
                ev_name = str(os.path.basename(str(ev))).split('.',1)[0]
                if ev_name in task_contrasts.keys() and os.stat(str(ev)).st_size:
                    ev_names.append(np.repeat(ev_name,len(pd.read_csv(ev,header = None).index)))
                    evs.append(str(ev))

            combined_csv = pd.concat( [ pd.read_csv(f,sep="\t",encoding="utf8",header = None,names=['onset','duration','rep']) for f in evs ] )
            combined_csv['task'] = list(itertools.chain(*ev_names))  #np.array(ev_names).flatten()
            combined_csv_sort = combined_csv.sort_values('onset')
            combined_csv_sort.to_csv( evfiles_combined, float_format='%.3f', sep='\t', encoding='utf-8', index=False,header=False)

            #events = pd.read_csv(evfiles_combined,sep="\t",encoding="utf8",header = None,names=['onset','duration','rep','task'])
    return None


def load_event_files(pathout,fmri_files,ev_filename=None):
    ###combining multiple sessions together
    '''
    tc_matrix = nib.load(fmri_files[0])
    Subject_Num = len(fmri_files)
    Trial_Num, Node_Num = tc_matrix.shape
    print("Data samples including %d subjects with %d trials and %d nodes for each" % (Subject_Num, Trial_Num, Node_Num))
    '''
    pathdata = Path(pathfmri)

    EVS_files = [];
    for ev,subcount in zip(sorted(pathdata.glob('*/FMRI/tfMRI_' + modality + '*/combined_events_spm_' + modality + '.csv')),np.arange(Subject_Num)):
        if os.path.dirname(fmri_files[subcount]) == os.path.dirname(str(ev)):
            EVS_files.append(str(ev))
        else:
            print("Event files and fmri data are miss-matching for subject: %s" % os.path.dirname(str(ev)))

    ###loading all event designs
    if not ev_filename:
        ev_filename = 'test.h5' #'.txt'

    events_all_subjects_file = pathout + modality + "_" + ev_filename
    if os.path.isfile(events_all_subjects_file):
        '''
        subjects_trial_labels = pd.read_csv(events_all_subjects_file, sep="\t", encoding="utf8")
        print(subjects_trial_labels.keys())

        subjects_trial_label_matrix = subjects_trial_labels.loc[:, 'trial1':'trial' + str(Trial_Num)]
        sub_name = subjects_trial_labels['subject']
        coding_direct = subjects_trial_labels['coding']
        '''
        subjects_trial_label_matrix = pd.read_hdf(events_all_subjects_file, 'trials')
        sub_name = pd.read_hdf(events_all_subjects_file, 'subject')
        coding_direct = pd.read_hdf(events_all_subjects_file, 'coding')
        print("Data samples including %d event files consisting of %d subjects, each has %d coding directions and %d trials"
              % (len(sub_name), len(np.unique(sub_name)), len(coding_direct)//len(np.unique(sub_name)), subjects_trial_label_matrix.shape[1]))
    else:
        subjects_trial_label_matrix = []
        sub_name = [];
        coding_direct = [];
        for subj in np.arange(Subject_Num):
            pathsub = Path(os.path.dirname(EVS_files[subj]))
            sub_name.append(pathsub.parts[-3])
            coding_direct.append(pathsub.parts[-1].split('_')[-1])

            ##trial info in volume
            trial_infos = pd.read_csv(EVS_files[subj], sep="\t", encoding="utf8", header=None,
                                      names=['onset', 'duration', 'rep', 'task'])
            Onsets = np.ceil((trial_infos.onset / TR)).astype(int)  # (trial_infos.onset/TR).astype(int)
            Duras = np.ceil((trial_infos.duration / TR)).astype(int)  # (trial_infos.duration/TR).astype(int)
            Movetypes = trial_infos.task

            labels = ["rest"] * Trial_Num
            for start, dur, move in zip(Onsets, Duras, Movetypes):
                for ti in range(start - 1, start + dur):
                    labels[ti] = task_contrasts[move]
            subjects_trial_label_matrix.append(labels)

        #print(np.array(subjects_trial_label_matrix).shape)
        #subjects_trial_label_matrix = np.array(subjects_trial_label_matrix)
        #sub_name = np.array(sub_name)
        #coding_direct = np.array(coding_direct)
        '''
        ### build a dataframe for evfiles
        subjects_trial_labels = pd.DataFrame(data=np.array(subjects_trial_label_matrix),
                                             columns=['trial' + str(i + 1) for i in range(Trial_Num)])
        subjects_trial_labels['subject'] = sub_name
        subjects_trial_labels['coding'] = coding_direct
        subjects_trial_labels.keys()
        subjects_trial_label_matrix = subjects_trial_labels.loc[:, 'trial1':'trial' + str(Trial_Num)]
        ##save the labels
        subjects_trial_labels.to_csv(events_all_subjects_file, sep='\t', encoding='utf-8', index=False)
        '''
        hdf = pd.HDFStore(events_all_subjects_file)
        # put the dataset in the storage
        subjects_trial_labels = pd.DataFrame(data=np.array(subjects_trial_label_matrix),
                                             columns=['trial' + str(i + 1) for i in range(Trial_Num)])
        hdf.put('trials', subjects_trial_labels, format='table', data_columns=True)
        hdf.append('subject', pd.DataFrame(np.array(sub_name), columns=['subj']), format='table', data_columns=True)
        hdf.append('coding', pd.DataFrame(np.array(coding_direct), columns=['code']), format='table', data_columns=True)
        hdf.close()  # closes the file

    return subjects_trial_label_matrix,sub_name,coding_direct,EVS_files


def load_event_files_tasktrials_only(pathout,fmri_files,ev_filename=None,window_size=None):
    ###combining multiple sessions together
    '''
    tc_matrix = nib.load(fmri_files[0])
    Subject_Num = len(fmri_files)
    Trial_Num, Node_Num = tc_matrix.shape
    print("Data samples including %d subjects with %d trials and %d nodes for each" % (Subject_Num, Trial_Num, Node_Num))
    '''
    pathdata = Path(pathfmri)

    EVS_files = []
    for ev,subcount in zip(sorted(pathdata.glob('*/FMRI/tfMRI_' + modality + '*/combined_events_spm_' + modality + '.csv')),np.arange(Subject_Num)):
        if os.path.dirname(fmri_files[subcount]) == os.path.dirname(str(ev)):
            EVS_files.append(str(ev))
        else:
            print("Event files and fmri data are miss-matching for subject: %s" % os.path.dirname(str(ev)))

    ###loading all event designs

    if not ev_filename:
        ev_filename = 'test.h5' #'.txt'
    if not window_size:
        window_str = '_tasktrial'
        window_size_trial = 1
    else:
        window_size_trial = math.floor(window_size / TR)
        if window_size_trial < 1:
            window_size_trial = 1
        window_str = '_tasktrial_win'+str(window_size)
    events_all_subjects_file = pathout + modality + window_str + "_" + ev_filename
    print(events_all_subjects_file)

    global Trial_Num
    trial_infos = pd.read_csv(EVS_files[0], sep="\t", encoding="utf8", header=None,
                              names=['onset', 'duration', 'rep', 'task'])
    Duras = np.ceil((trial_infos.duration / TR)).astype(int)
    Trial_Num = np.sum(np.floor(Duras / window_size_trial).astype(int))
    if os.path.isfile(events_all_subjects_file):
        '''
        subjects_trial_labels = pd.read_csv(events_all_subjects_file, sep="\t", encoding="utf8")
        print(subjects_trial_labels.keys())

        subjects_trial_label_matrix = subjects_trial_labels.loc[:, 'trial1':'trial' + str(Trial_Num)]
        sub_name = subjects_trial_labels['subject']
        coding_direct = subjects_trial_labels['coding']
        '''
        subjects_trial_label_matrix = pd.read_hdf(events_all_subjects_file, 'trials')
        sub_name = pd.read_hdf(events_all_subjects_file, 'subject')
        coding_direct = pd.read_hdf(events_all_subjects_file, 'coding')
        #print("Data samples including %d event files consisting of %d subjects, each has %d coding directions and %d trials"
        #      % (len(sub_name), len(np.unique(sub_name)), len(coding_direct) // len(np.unique(sub_name)),subjects_trial_label_matrix.shape[1]))
    else:
        subjects_trial_label_matrix = []
        sub_name = []
        coding_direct = []
        for subj in np.arange(Subject_Num):
            pathsub = Path(os.path.dirname(EVS_files[subj]))
            sub_name.append(pathsub.parts[-3])
            coding_direct.append(pathsub.parts[-1].split('_')[-1])

            ##trial info in volume
            trial_infos = pd.read_csv(EVS_files[subj], sep="\t", encoding="utf8", header=None,
                                      names=['onset', 'duration', 'rep', 'task'])
            Onsets = np.ceil((trial_infos.onset / TR)).astype(int)  # (trial_infos.onset/TR).astype(int)
            Duras = np.ceil((trial_infos.duration / TR)).astype(int)  # (trial_infos.duration/TR).astype(int)
            Movetypes = trial_infos.task

            labels = ["rest"] * Trial_Num
            start = 0
            for dur, move in zip(Duras, Movetypes):
                dur_wind = np.floor(dur / window_size_trial).astype(int)
                if dur_wind < 1:
                    dur_wind = 1
                for ti in range(start, start + dur_wind):
                    labels[ti] = task_contrasts[move]
                start += dur_wind
            subjects_trial_label_matrix.append(labels)

        #print(np.array(subjects_trial_label_matrix).shape)
        #subjects_trial_label_matrix = np.array(subjects_trial_label_matrix)
        #sub_name = np.array(sub_name)
        #coding_direct = np.array(coding_direct)
        '''
        ### build a dataframe for evfiles
        subjects_trial_labels = pd.DataFrame(data=np.array(subjects_trial_label_matrix),
                                             columns=['trial' + str(i + 1) for i in range(Trial_Num)])
        subjects_trial_labels['subject'] = sub_name
        subjects_trial_labels['coding'] = coding_direct
        subjects_trial_labels.keys()
        subjects_trial_label_matrix = subjects_trial_labels.loc[:, 'trial1':'trial' + str(Trial_Num)]
        ##save the labels
        subjects_trial_labels.to_csv(events_all_subjects_file, sep='\t', encoding='utf-8', index=False)
        '''
        hdf = pd.HDFStore(events_all_subjects_file)
        # put the dataset in the storage
        subjects_trial_label_matrix = pd.DataFrame(data=np.array(subjects_trial_label_matrix),
                                                   columns=['trial' + str(i + 1) for i in range(Trial_Num)])
        hdf.put('trials', subjects_trial_label_matrix, format='table', data_columns=True)
        hdf.append('subject', pd.DataFrame(np.array(sub_name), columns=['subj']), format='table', data_columns=True)
        hdf.append('coding', pd.DataFrame(np.array(coding_direct), columns=['code']), format='table', data_columns=True)
        hdf.close()  # closes the file

    print("Data samples including %d event files consisting of %d subjects, each has %d coding directions and %d trials"
          % (len(sub_name), len(np.unique(sub_name)), len(coding_direct) // len(np.unique(sub_name)),subjects_trial_label_matrix.shape[1]))

    return subjects_trial_label_matrix,sub_name,coding_direct,EVS_files


def map_surface_to_parcel(parcel_map, fmri_file):

    atlas_roi = nib.load(parcel_map).get_data()
    #RegionLabels = np.unique(atlas_roi)
    RegionLabels = [i for i in np.unique(atlas_roi) if i != 0]

    tc_matrix = nib.load(fmri_file).get_data()
    if atlas_roi.shape[1] != tc_matrix.shape[1]:
        print("Dimension of parcellation map not matching")
        print("%d vs %d " % (atlas_roi.shape[1], tc_matrix.shape[1]))
        tc_matrix = tc_matrix[:, range(atlas_roi.shape[1])]

    tc_roi_matrix = []
    for li in sorted(RegionLabels):
        tmp_ind = [ind for ind in range(tc_matrix.shape[1]) if atlas_roi[0][ind] == li]
        tc_roi_matrix.append(np.mean(tc_matrix[:, tmp_ind], axis=1))

    ##print(np.transpose(np.array(tc_roi_matrix),(1,0)).shape)
    return tc_roi_matrix


#######################################
####tensorpack: multithread
class dataflow_fmri_with_confound(dataflow.DataFlow):
    """ Iterate through fmri filenames and confound filenames
    """

    def __init__(self, fmri_files, confound_files):
        assert (len(fmri_files) == len(confound_files))
        # self.data=zip(fmri_files,confound_files)
        self.fmri_files = fmri_files
        self.confound_files = confound_files
        self._size = len(fmri_files)

    def size(self):
        return self._size

    def get_data(self):
        for a, b in zip(self.fmri_files, self.confound_files):
            yield a, b


def map_func_extract_seris(dp, mmp_atlas_roi):
    fmri_file = dp[0]
    confound_file = dp[1]

    tc_roi_matrix = map_surface_to_parcel(mmp_atlas_roi, fmri_file)
    tc_matrix = np.transpose(np.array(tc_roi_matrix), (1, 0))
    confound = np.loadtxt(confound_file)
    ##regress out confound effects
    tc_matrix = preprocessing.scale(tc_matrix)
    regr.fit(confound, tc_matrix)
    mist_roi_tc = tc_matrix - np.matmul(confound, regr.coef_.T)

    return [fmri_file, mist_roi_tc]


def dump_mean_seris_to_lmdb(df, lmdb_path, write_frequency=10):
    """
    save extracted series into lmdb:lmdb_path
    """
    assert isinstance(df, dataflow.DataFlow), type(df)
    isdir = os.path.isdir(lmdb_path)
    df.reset_state()
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)  # need sync() at the end
    try:
        sz = df.size()
    except NotImplementedError:
        sz = 0

    with get_tqdm(total=sz) as pbar:
        idx = -1
        keys = []
        txn = db.begin(write=True)
        for idx, dp in enumerate(df.get_data()):
            txn.put(u'{}'.format(dp[0]).encode('ascii'), dumps(dp[1]))
            keys += [u'{}'.format(dp[0]).encode('ascii')]
            pbar.update()
            if (idx + 1) % write_frequency == 0:
                txn.commit()
                txn = db.begin(write=True)
        txn.commit()

        with db.begin(write=True) as txn:
            txn.put(b'__keys__', dumps(keys))

        logger.info("Flushing database ...")
        db.sync()
    db.close()


def extract_mean_seris(fmri_files, confound_files, mmp_atlas_roi, lmdb_filename, nr_thread=25, buffer_size=50):
    '''extract roi mean time series and save to lmdb file
    '''
    ds0 = dataflow_fmri_with_confound(fmri_files, confound_files)
    print('dataflowSize is ' + str(ds0.size()))

    print('buffer_size is ' + str(buffer_size))

    ds1 = dataflow.MultiThreadMapData(
        ds0, nr_thread=nr_thread,
        map_func=lambda dp: map_func_extract_seris(dp, mmp_atlas_roi),
        buffer_size=buffer_size,
        strict=True)
    ds1 = dataflow.PrefetchDataZMQ(ds1, nr_proc=1)
    ds1._reset_once()

    dump_mean_seris_to_lmdb(ds1, lmdb_filename)
    
###end of tensorpack: multithread
##############################################################


def load_fmri_matrix_from_atlas(pathout,mmp_atlas,fmri_files,confound_files=None,fmri_matrix_name=None):
    ##loading fMRI signals using atlas mapping

    mmp_atlas_roi = nib.load(mmp_atlas).get_data()
    global Region_Num
    Region_Num = len(np.unique(mmp_atlas_roi))
    print("%d regions in the parcellation map" % Region_Num)

    if not fmri_matrix_name:
        fmri_matrix_name = 'test.h5' #'test.txt'
    tcs_all_subjects_file = pathout + modality + "_" + fmri_matrix_name

    if os.path.isfile(tcs_all_subjects_file):
        ###loading time-series data from files
        '''
        subjects_tc_matrix = np.loadtxt(tcs_all_subjects_file).reshape((Subject_Num, Trial_Num, Region_Num))
        '''
        hdf = h5py.File(tcs_all_subjects_file, 'r')
        group1 = hdf.get('region_tc')
        group2 = hdf.get('sub_info')

        subjects_tc_matrix = []
        for subj in range(Subject_Num):
            n1 = np.array(group1.get('subj' + str(subj + 1)))
            subjects_tc_matrix.append(n1)
        sub_name = [sub.decode("utf-8") for sub in np.array(group2.get('sub_name'))]
        coding_direct = [sub.decode("utf-8") for sub in np.array(group2.get('coding'))]
        hdf.close()
        print(np.array(subjects_tc_matrix).shape)
    else:
        ###read fmri data through atlas mapping
        if not confound_files:
            confound_files = [''] * len(fmri_files)

        subjects_tc_matrix = []
        sub_name = []
        coding_direct = []
        for fmri_file, confound_file,sub_count in zip(fmri_files, confound_files,np.arange(0,Subject_Num)):
            pathsub = Path(os.path.dirname(fmri_files[sub_count]))
            sub_name.append(pathsub.parts[-3])
            coding_direct.append(pathsub.parts[-1].split('_')[-1])

            ##step1:load fmri time-series and confounds into matrix
            tc_roi_matrix = map_surface_to_parcel(mmp_atlas, fmri_file)
            tc_matrix = np.transpose(np.array(tc_roi_matrix), (1, 0))
            if confound_file == '':
                confound = np.ones((tc_matrix.shape[0],), dtype=int)
            else:
                confound = np.loadtxt(confound_file)

            ##step2: regress out confound effects and save the residuals
            tc_matrix = preprocessing.scale(tc_matrix)
            regr.fit(confound, tc_matrix)
            mist_roi_tc = tc_matrix - np.matmul(confound, regr.coef_.T)

            subjects_tc_matrix.append(mist_roi_tc)
            if divmod(sub_count, 100)[1] == 0:
                print("Processing subjects: %d" % sub_count)

        print(np.array(subjects_tc_matrix).shape)
        #subjects_tc_matrix = np.array(subjects_tc_matrix)
        #sub_name = np.array(sub_name)
        #coding_direct = np.array(coding_direct)

        hdf5 = h5py.File(tcs_all_subjects_file, 'w')
        g1 = hdf5.create_group('region_tc')
        # g1.create_dataset('subjects',data=np.array(subjects_tc_matrix),compression="gzip")
        for subj in range(Subject_Num):
            g1.create_dataset('subj' + str(subj + 1), data=subjects_tc_matrix[subj], compression="gzip")
        g2 = hdf5.create_group('sub_info')
        g2.create_dataset('sub_name', data=np.array(sub_name).astype('|S9'), compression="gzip")
        g2.create_dataset('coding', data=np.array(coding_direct).astype('|S9'), compression="gzip")
        hdf5.close()  # closes the file

        '''
        # Write the array to disk
        with open(tcs_all_subjects_file, 'w') as outfile:
            outfile.write('# Array shape: {0}\n'.format(np.array(subjects_tc_matrix).shape))
            for data_slice in subjects_tc_matrix:
                np.savetxt(outfile, data_slice, fmt='%-7.2f')
                outfile.write('# New slice\n')
        '''

    return subjects_tc_matrix,sub_name,coding_direct


def load_fmri_matrix_from_atlas_tasktrials_only(pathout,mmp_atlas,fmri_files,EVS_files,confound_files=None,fmri_matrix_name=None,window_size=None):
    ##loading fMRI signals within task trials using atlas mapping

    mmp_atlas_roi = nib.load(mmp_atlas).get_data()
    global Region_Num
    Region_Num = len(np.unique(mmp_atlas_roi))
    print("%d regions in the parcellation map" % Region_Num)

    if not window_size:
        window_str = '_tasktrial'
        window_size_trial = 1
    else:
        window_size_trial = math.floor(window_size / TR)
        if window_size_trial < 1:
            window_size_trial = 1
        window_str = '_tasktrial_win'+str(window_size)
    if not fmri_matrix_name:
        fmri_matrix_name = 'test.h5' #'test.txt'
    tcs_all_subjects_file = pathout + modality + window_str + "_" + fmri_matrix_name

    if os.path.isfile(tcs_all_subjects_file):
        ###loading time-series data from files
        '''
        subjects_tc_matrix = np.loadtxt(tcs_all_subjects_file).reshape((Subject_Num, Trial_Num, Region_Num))
        '''
        hdf = h5py.File(tcs_all_subjects_file, 'r')
        group1 = hdf.get('region_tc')
        group2 = hdf.get('sub_info')

        subjects_tc_matrix = []
        for subj in range(Subject_Num):
            n1 = np.array(group1.get('subj'+str(subj+1)))
            subjects_tc_matrix.append(n1)
        sub_name = [sub.decode("utf-8") for sub in np.array(group2.get('sub_name'))]
        coding_direct = [sub.decode("utf-8") for sub in np.array(group2.get('coding'))]
        hdf.close()
        print(np.array(subjects_tc_matrix).shape)
    else:
        ###read fmri data through atlas mapping
        if not confound_files:
            confound_files = [''] * len(fmri_files)

        subjects_tc_matrix = []
        sub_name = []
        coding_direct = []
        for fmri_file, confound_file, ev_file, sub_count in zip(fmri_files, confound_files,EVS_files, np.arange(0,Subject_Num)):
            pathsub = Path(os.path.dirname(fmri_files[sub_count]))
            sub_name.append(pathsub.parts[-3])
            coding_direct.append(pathsub.parts[-1].split('_')[-1])
            ##step1:load fmri time-series and confounds into matrix
            tc_roi_matrix = map_surface_to_parcel(mmp_atlas, fmri_file)
            tc_matrix = np.transpose(np.array(tc_roi_matrix), (1, 0))
            if confound_file == '':
                confound = np.ones((tc_matrix.shape[0],), dtype=int)
            else:
                confound = np.loadtxt(confound_file)

            ##step2: regress out confound effects and save the residuals
            tc_matrix = preprocessing.scale(tc_matrix)
            regr.fit(confound, tc_matrix)
            mist_roi_tc = tc_matrix - np.matmul(confound, regr.coef_.T)

            ##step3: extract only task trial tc
            trial_infos = pd.read_csv(ev_file, sep="\t", encoding="utf8", header=None,
                                      names=['onset', 'duration', 'rep', 'task'])
            Onsets = np.ceil((trial_infos.onset / TR)).astype(int)  # (trial_infos.onset/TR).astype(int)
            Duras = np.ceil((trial_infos.duration / TR)).astype(int)  # (trial_infos.duration/TR).astype(int)

            tc_matrix_tasktrial = []
            for start, dur in zip(Onsets, Duras):
                dur_wind = np.floor(dur / window_size_trial).astype(int)
                fmri_tc_trial = []
                if dur_wind < 1:
                    fmri_tc_trial.append(np.mean(mist_roi_tc[start:start + dur, :], axis=0))
                for di in np.arange(dur_wind):
                    start_wind = start + di * window_size_trial
                    end_wind = start + (di + 1) * window_size_trial
                    if end_wind >= start + dur:
                        end_wind = start + dur
                    # print(start_wind,end_wind,start + dur)
                    fmri_tc_trial.append(np.mean(mist_roi_tc[start_wind:end_wind, :], axis=0))
                # fmri_tc_trial = mist_roi_tc[start:start + dur]  ##extract task trials
                tc_matrix_tasktrial.append(fmri_tc_trial)

            subjects_tc_matrix.append(np.vstack(tc_matrix_tasktrial))
            if divmod(sub_count, 100)[1] == 0:
                print("Processing subjects: %d" % sub_count)

        print(np.array(subjects_tc_matrix).shape)
        #subjects_tc_matrix = np.array(subjects_tc_matrix)
        #sub_name = np.array(sub_name)
        #coding_direct = np.array(coding_direct)

        hdf5 = h5py.File(tcs_all_subjects_file, 'w')
        g1 = hdf5.create_group('region_tc')
        # g1.create_dataset('subjects',data=np.array(subjects_tc_matrix),compression="gzip")
        for subj in range(Subject_Num):
            g1.create_dataset('subj' + str(subj + 1), data=subjects_tc_matrix[subj], compression="gzip")
        g2 = hdf5.create_group('sub_info')
        g2.create_dataset('sub_name', data=np.array(sub_name).astype('|S9'), compression="gzip")
        g2.create_dataset('coding', data=np.array(coding_direct).astype('|S9'), compression="gzip")
        hdf5.close()  # closes the file

        '''
        # Write the array to disk
        with open(tcs_all_subjects_file, 'w') as outfile:
            outfile.write('# Array shape: {0}\n'.format(np.array(subjects_tc_matrix).shape))
            for data_slice in subjects_tc_matrix:
                np.savetxt(outfile, data_slice, fmt='%-7.2f')
                outfile.write('# New slice\n')
        '''
    return subjects_tc_matrix,sub_name,coding_direct



def plot_roc_curve(X_train, Y_train_int, X_test, Y_test_int):
    ##roc curve
    from sklearn import preprocessing
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn import svm, metrics

    n_classes = len(np.unique(Y_train_int))
    Y_train_mat = preprocessing.label_binarize(Y_train_int, classes=np.unique(Y_train_int))
    Y_test_mat = preprocessing.label_binarize(Y_test_int, classes=np.unique(Y_train_int))

    clf = OneVsRestClassifier(svm.SVC(kernel='linear', decision_function_shape='ovr'))
    y_score = clf.fit(X_train, Y_train_mat).decision_function(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(Y_test_mat[:, i], y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    lw = 2
    '''
    task_select = 0
    plt.figure()
    plt.plot(fpr[task_select], tpr[task_select], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[task_select])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    '''

    ###Plot ROC curves for the multiclass
    # First aggregate all false positive rates
    from scipy import interp
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(Y_test_mat.ravel(), y_score.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    from itertools import cycle
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    return None


def classification_accuarcy(X_data_pca, Y_data, my_testsize=0.2, target_name=None):
    from sklearn import svm, metrics
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    n_classes = len(np.unique(Y_data))
    if target_name is None:
        target_name = range(n_classes)

    X_train, X_test, Y_train, Y_test = train_test_split(X_data_pca, Y_data, test_size=my_testsize, random_state=10)
    clf = svm.SVC(kernel='linear', decision_function_shape='ovo')
    clf.fit(X_train, Y_train)
    acc = metrics.accuracy_score(clf.predict(X_test), Y_test)
    # print(acc)

    ##f1 score for multiclass
    le = preprocessing.LabelEncoder()
    le.fit(target_name)
    Y_train_int = le.transform(Y_train)
    Y_test_int = le.transform(Y_test)
    # print(np.unique(Y_train_int))


    clf.fit(X_train, Y_train_int)
    f1score = metrics.f1_score(clf.predict(X_test), Y_test_int, average='macro')
    print('Accuarcy on test data: %4f and f1-score %4f' % (acc, f1score))
    y_pred = clf.predict(X_test)
    print(metrics.classification_report(Y_test_int, y_pred, target_names=target_name))
    print('Confusion Matrix:')
    print(metrics.confusion_matrix(Y_test_int, y_pred, labels=range(n_classes)))

    plot_roc_curve(X_train, Y_train_int, X_test, Y_test_int)
    return None


def my_svc_simple(subjects_tc_matrix,subjects_trial_label_matrix,target_name,my_cv_fold=10,my_comp=20):
    ###SVM classifier
    ##feature matrix
    fmri_data = np.vstack(subjects_tc_matrix)
    #label_data = subjects_trial_label_matrix[:Subject_Num].as_matrix().reshape(Subject_Num*Trial_Num,)
    label_data = np.array(subjects_trial_label_matrix[:Subject_Num]).reshape(Subject_Num * Trial_Num, )
    print("%d Data samples with %d features for each" % (fmri_data.shape[0], fmri_data.shape[1]))

    #target_name = np.unique(list(task_contrasts.values()))
    condition_mask = pd.Series(label_data).isin(target_name)
    X_data = fmri_data[condition_mask,]
    Y_data = label_data[condition_mask]

    ##build a simple classifier using SVM
    X_data_scaled = preprocessing.scale(X_data) #with zero mean and unit variance.
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(X_data_scaled, Y_data)
    acc_score = metrics.accuracy_score(clf.predict(X_data_scaled), Y_data)
    print("Accuracy of prediction with SVM-RBF kernel: %4f" % acc_score)

    ##cross-validation
    X_data_scaled = preprocessing.scale(X_data)
    scores = cross_val_score(clf, X_data_scaled, Y_data, cv=my_cv_fold, scoring='accuracy')
    print('SVM Scoring:')
    print(scores)

    ##using pca for dimension reduction
    pca = PCA(n_components=my_comp, svd_solver='randomized', whiten=True)
    pca.fit(X_data_scaled)
    X_data_pca = pca.fit_transform(X_data_scaled)
    scores_pca = cross_val_score(clf, X_data_pca, Y_data, cv=my_cv_fold, scoring='accuracy')
    print('SVM Scoring after PCA decomposition: ')
    print(scores_pca)

    ##using fastica for dimension reduction
    ica = FastICA(n_components=20, whiten=True)
    ica.fit(X_data_scaled)
    X_data_ica = ica.fit_transform(X_data_scaled)
    scores_ica = cross_val_score(clf, X_data_ica, Y_data, cv=5, scoring='accuracy')
    print('SVM Scoring after ICA decomposition: ')
    print(scores_ica)

    ##using kernelPCA for dimension reduction
    kpca = KernelPCA(n_components=my_comp, kernel='rbf')
    kpca.fit(X_data_scaled)
    X_data_kpca = kpca.fit_transform(X_data_scaled)
    scores_kpca = cross_val_score(clf, X_data_kpca, Y_data, cv=my_cv_fold, scoring='accuracy')
    print('SVM Scoring after DL decomposition: ')
    print(scores_kpca)
    return scores, scores_pca, scores_ica, scores_kpca


def subject_cross_validation_split(tc_matrix, label_matrix, n_folds=20, testsize=0.2, valsize=0.2, randomseed=1234):
    Subject_Num, Trial_Num = np.array(subjects_trial_label_matrix).shape
    rs = np.random.RandomState(randomseed)
    train_sid_tmp, test_sid = train_test_split(range(Subject_Num), test_size=testsize, random_state=rs, shuffle=True)
    fmri_data_train = np.array(np.vstack([tc_matrix[i] for i in train_sid_tmp]))
    fmri_data_test = np.array(np.vstack([tc_matrix[i] for i in test_sid]))

    label_data_train = np.array([label_matrix[i] for i in train_sid_tmp])
    label_data_test = np.array([label_matrix[i] for i in test_sid]).reshape(len(test_sid) * Trial_Num, )

    scaler = preprocessing.StandardScaler().fit(fmri_data_train)
    X_train = scaler.transform(fmri_data_train).reshape(len(train_sid_tmp), Trial_Num, Region_Num)
    X_test = scaler.transform(fmri_data_test)
    Y_train = label_data_train
    Y_test = label_data_test
    nb_class = len(np.unique(label_matrix))

    from sklearn.model_selection import ShuffleSplit
    valsplit = ShuffleSplit(n_splits=n_folds, test_size=valsize, random_state=rs)
    X_train_scaled = []
    X_val_scaled = []
    Y_train_scaled = []
    Y_val_scaled = []
    for train_sid, val_sid in valsplit.split(train_sid_tmp):
        ##preprocess features and labels
        X_train_scaled.append(np.array(np.vstack([X_train[i] for i in train_sid])))
        X_val_scaled.append(np.array(np.vstack([X_train[i] for i in val_sid])))
        Y_train_scaled.append(np.ravel(np.array(np.vstack([Y_train[i] for i in train_sid]))))
        Y_val_scaled.append(np.ravel(np.array(np.vstack([Y_train[i] for i in val_sid]))))

        '''
        # print(X_train_scaled.shape,Y_train_scaled.shape,X_val_scaled.shape)
        clf = svm.SVC(decision_function_shape='ovo')
        clf.fit(X_train_scaled, Y_train_scaled)
        train_acc.append(metrics.accuracy_score(clf.predict(X_train_scaled), Y_train_scaled))
        val_acc.append(metrics.accuracy_score(clf.predict(X_val_scaled), Y_val_scaled))
        test_acc.append(metrics.accuracy_score(clf.predict(X_test), Y_test))
    print("Accuracy of prediction with SVM-RBF kernel in training:{},validation:{} and testing:{}"
          .format(np.mean(train_acc), np.mean(val_acc), np.mean(test_acc)))
          '''
    print('Samples for training: %d and testing %d and validating %d with %d classes' % (len(train_sid), len(test_sid), len(val_sid), nb_class))
    return X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled, X_test, Y_test


def my_svc_simple_subject_validation(subjects_tc_matrix,subjects_trial_label_matrix,my_cv_fold=10,my_comp=20,my_testsize=0.2,my_valsize=0.2):
    ###SVM classifier
    ##split data into train, val and test in subject-level
    X_train, Y_train, X_val, Y_val, X_test, Y_test = subject_cross_validation_split(subjects_tc_matrix, np.array(subjects_trial_label_matrix),
                                                                                    n_folds=my_cv_fold, testsize=my_testsize, valsize=my_valsize)

    ##build a simple classifier using SVM
    clf = svm.SVC(decision_function_shape='ovo')
    train_acc = []
    val_acc = []
    for x_train, y_train, x_val, y_val in zip(X_train, Y_train, X_val, Y_val):
        clf.fit(x_train, y_train)
        train_acc.append(metrics.accuracy_score(clf.predict(x_train), y_train))
        val_acc.append(metrics.accuracy_score(clf.predict(x_val), y_val))
    test_acc = metrics.accuracy_score(clf.predict(X_test), Y_test)
    scores = [np.mean(train_acc),np.mean(val_acc),test_acc]
    print('SVM Scoring:')
    print("Accuracy of prediction in training:{},validation:{} and testing:{}"
          .format(scores[0], scores[1], scores[2]))


    ##using pca for dimension reduction
    X_data_scaled = np.vstack([X_train[0], X_val[0]])
    pca = PCA(n_components=my_comp, svd_solver='randomized', whiten=True)
    pca.fit(X_data_scaled)
    train_acc = []
    val_acc = []
    for x_train, y_train, x_val, y_val in zip(X_train, Y_train, X_val, Y_val):
        x_train_pca = pca.fit_transform(x_train)
        x_val_pca = pca.fit_transform(x_val)
        clf.fit(x_train_pca, y_train)
        train_acc.append(metrics.accuracy_score(clf.predict(x_train_pca), y_train))
        val_acc.append(metrics.accuracy_score(clf.predict(x_val_pca), y_val))
    X_test_pca = pca.fit_transform(X_test)
    test_acc = metrics.accuracy_score(clf.predict(X_test_pca), Y_test)
    scores_pca = [np.mean(train_acc),np.mean(val_acc),test_acc]
    print('SVM Scoring after PCA decomposition: ')
    print("Accuracy of prediction in training:{},validation:{} and testing:{}"
          .format(scores_pca[0], scores_pca[1], scores_pca[2]))

    ##using fastica for dimension reduction
    ica = FastICA(n_components=my_comp, whiten=True)
    ica.fit(X_data_scaled)
    train_acc = []
    val_acc = []
    for x_train, y_train, x_val, y_val in zip(X_train, Y_train, X_val, Y_val):
        x_train_ica = ica.fit_transform(x_train)
        x_val_ica = ica.fit_transform(x_val)
        clf.fit(x_train_ica, y_train)
        train_acc.append(metrics.accuracy_score(clf.predict(x_train_ica), y_train))
        val_acc.append(metrics.accuracy_score(clf.predict(x_val_ica), y_val))
    X_test_ica = ica.fit_transform(X_test)
    test_acc = metrics.accuracy_score(clf.predict(X_test_ica), Y_test)
    scores_ica = [np.mean(train_acc), np.mean(val_acc), test_acc]
    print('SVM Scoring after ICA decomposition: ')
    print("Accuracy of prediction in training:{},validation:{} and testing:{}"
          .format(scores_ica[0], scores_ica[1], scores_ica[2]))

    ##using kernelPCA for dimension reduction
    kpca = KernelPCA(n_components=my_comp, kernel='rbf')
    kpca.fit(X_data_scaled)
    train_acc = []
    val_acc = []
    for x_train, y_train, x_val, y_val in zip(X_train, Y_train, X_val, Y_val):
        x_train_kpca = kpca.fit_transform(x_train)
        x_val_kpca = kpca.fit_transform(x_val)
        clf.fit(x_train_kpca, y_train)
        train_acc.append(metrics.accuracy_score(clf.predict(x_train_kpca), y_train))
        val_acc.append(metrics.accuracy_score(clf.predict(x_val_kpca), y_val))
    X_test_kpca = kpca.fit_transform(X_test)
    test_acc = metrics.accuracy_score(clf.predict(X_test_kpca), Y_test)
    scores_kpca = [np.mean(train_acc), np.mean(val_acc), test_acc]
    print('SVM Scoring after DL decomposition: ')
    print("Accuracy of prediction in training:{},validation:{} and testing:{}"
          .format(scores_kpca[0], scores_kpca[1], scores_kpca[2]))

    return scores, scores_pca, scores_ica, scores_kpca


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


def build_fc_nn_simple(subjects_tc_matrix,subjects_trial_label_matrix,target_name,layers=3,hidden_size=256,dropout=0.25):
    ###classification using fully-connected neural networks

    ##feature matrix
    fmri_data = np.vstack(subjects_tc_matrix)
    label_data = subjects_trial_label_matrix[:Subject_Num].values.reshape(Subject_Num * Trial_Num, )
    #target_name = np.unique(list(task_contrasts.values()))  ##whether to exclude 'rest'
    condition_mask = pd.Series(label_data).isin(target_name)
    X_data = fmri_data[condition_mask,]
    Y_data = label_data[condition_mask]

    X_data_scaled = preprocessing.scale(X_data)  # with zero mean and unit variance.
    le = preprocessing.LabelEncoder()
    le.fit(target_name)
    Y_data_int = le.transform(Y_data)
    Y_data_label = np_utils.to_categorical(Y_data_int)

    print("%d Data samples with %d features and output in %d classes" %
          (X_data_scaled.shape[0],X_data_scaled.shape[1], Y_data_label.shape[1]))

    ######fully-connected neural networks
    Nfeatures = X_data_scaled.shape[1]
    input0 = Input(shape=(Nfeatures,))
    drop1 = input0
    for li in np.arange(layers):
        hidden1 = Dense(hidden_size, activation='relu')(drop1)
        drop1 = Dropout(dropout)(hidden1)
        hidden_size = np.int32(hidden_size / 2)
        if hidden_size < 10:
            hidden_size = 16

    hidden2 = Dense(16, activation='relu')(drop1)
    drop2 = Dropout(0.5)(hidden2)
    out = Dense(len(target_name), activation='softmax')(drop2)

    model = Model(inputs=input0, outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mse','accuracy'])
    model.summary()

    ####running the model
    start_time = time.clock()
    X_train, X_test, Y_train, Y_test = train_test_split(X_data_scaled, Y_data_label, test_size=0.1, random_state=10)
    print('Samples for training: %d and testing %d with %d features' % (
    X_train.shape[0], X_test.shape[0], X_train.shape[1]))

    model_history = model.fit(X_train, Y_train, batch_size=128, epochs=100, validation_split=0.1)
    ##plot_history(model_history)
    print('Time Usage in seconds: {}'.format(time.clock() - start_time))
    train_loss = model_history.history['loss'][-1]
    train_acc = model_history.history['acc'][-1]
    val_loss = model_history.history['val_loss'][-1]
    val_acc = model_history.history['val_acc'][-1]

    test_loss, test_mse, test_acc = model.evaluate(X_test, Y_test)
    print('Testing sets: Loss: %4f and Accuracy: %4f' % (test_loss, test_acc))
    return test_acc, val_acc, train_acc


if __name__ == '__main__':
    args = sys.argv[1:]
    logger.set_logger_dir("train_log/svc_simple_log.txt",action="d")

    parser = argparse.ArgumentParser(description='The description of the parameters')

    parser.add_argument('--fmri_folder', '-a', help='(required, string) Path to the fMRI data for each subject under FMRI subfolder', type=str)
    parser.add_argument('--temp_outputdir', '-b', help='(required, string) Path to the temporary output including fmri tc-matrix and label-txt', type=str)
    parser.add_argument('--task_modality', '-c', help='(required, string) Modality name in Capital for fmri and event design files', type=str)
    parser.add_argument('--atlas_name', '-d', help='(required, string) Name of the Atlas for mapping fmri data', type=str)
    parser.add_argument('--atlas_filename', '-e', help='(required, string) Atlas filename for mapping fmri tc', type=str)

    parser.add_argument('--subject_to_start', '-f', help='(optional, int,default=0) The index of the first subject in the all_subjects_list for analysis', type=int)
    parser.add_argument('--subject_to_last', '-g', help='(optional, int,default=1086) The index of the last subject in the all_subjects_list for analysis', type=int)
    parser.add_argument('--subjectlist_index', '-l', help='(optional, string, default='') The index indicator of the selected subject list', type=str)
    parser.add_argument('--window_size_tasktrial', '-w', help='(optional, int, default=0) The size of window during extraction of fmri data and task labels', type=int)

    parser.add_argument('--step_indicator', '-s', help='(optional, int,default=0) The indicator of how far of data analysis, 0 for stetup, 1 for loading data, 2 for classification ', type=int)
    parser.add_argument('--max_iter', '-v', help='(optional, int, default = 80) Maximum number of iterations of the k-means algorithm for a single run.', type= int)
    parser.add_argument('--n_sessions', '-j', help='(optional, int, default = 0)  Total number of session for the subject', type=int)
    parser.add_argument('--n_sessions_combined', '-x', help='(optional, int, default = 1) The number of sessions to combine', type=int)

    parsed, unknown = parser.parse_known_args(args)
    ##global pathfmri, pathout, modality, mmp_atlas,AtlasName
    pathfmri = parsed.fmri_folder
    pathout = parsed.temp_outputdir
    modality = parsed.task_modality
    AtlasName = parsed.atlas_name
    mmp_atlas = parsed.atlas_filename
    startsub = parsed.subject_to_start
    endsub = parsed.subject_to_last
    subjectlist = parsed.subjectlist_index
    window_size = parsed.window_size_tasktrial
    flag_steps = parsed.step_indicator

    n_jobs = 1
    max_iter = parsed.max_iter
    n_sessions = parsed.n_sessions
    n_sessions_combined = parsed.n_sessions_combined
    if not flag_steps:
        flag_steps = 0

    ###output logs
    print("--fmri_folder: ", pathfmri)
    print('--temp_out:', pathout)
    print('--atlas_filename:', mmp_atlas)

    fmri_filename = 'Atlas.dtseries.nii'
    confound_filename = 'Movement_Regressors.txt'


    ##pre-settings of the analysis
    task_contrasts = bulid_dict_task_modularity(modality)
    target_name = np.unique(list(task_contrasts.values()))
    print('Processing with the task_modality: %s with %d subtypes' % (modality, len(target_name)))
    print(target_name)

    ##collecting the fmri files
    fmri_files, confound_files = load_fmri_data(pathfmri, modality,postfix= fmri_filename,confound_name= confound_filename)
    print('Collecting fmri files from subject {} to {}'.format(startsub, endsub))
    print('Generating Subject List of %s' % subjectlist)
    if Subject_Num < 10:
        print(fmri_files)

    ##collecting the event design files and loading fmri data
    ev_filename = 'event_labels_1200R'+'_test_'+subjectlist+'.h5' #'.txt'
    fmri_matrix_filename = AtlasName + '_ROI_act_1200R' + '_test_' + subjectlist + '.h5' #'.txt'
    combine_events_spm(fmri_files)


    if flag_steps >= 1:
        if not window_size:
            print('Use fmri and label data from all trials in the following analysis!!!')

            print('Collecting event design files for subjects and saved into matrix ...')
            subjects_trial_label_matrix,sub_name,coding,EVS_files = load_event_files(pathout,fmri_files,ev_filename=ev_filename)

            ##loading fmri data
            print('Collecting fmri time-series for subjects and saved into matrix ...')
            subjects_tc_matrix,sub_name,coding = load_fmri_matrix_from_atlas(pathout, mmp_atlas, fmri_files, confound_files=confound_files,
                                                                             fmri_matrix_name=fmri_matrix_filename)
        else:
            print('Use fmri and label data only from task trials with %d window-size in the following analysis!!!' % window_size)

            print('Collecting event design files for subjects and saved into matrix ...')
            subjects_trial_label_matrix,sub_name,coding,EVS_files = load_event_files_tasktrials_only(pathout, fmri_files, ev_filename=ev_filename,
                                                                                                     window_size=window_size)

            ##loading fmri data
            print('Collecting fmri time-series for subjects and saved into matrix ...')
            subjects_tc_matrix,sub_name,coding = \
                load_fmri_matrix_from_atlas_tasktrials_only(pathout, mmp_atlas, fmri_files, EVS_files, confound_files=confound_files,
                                                            fmri_matrix_name=fmri_matrix_filename,window_size=window_size)
        Subject_Names = np.unique(sub_name)

        if flag_steps >= 2:
            ##classifier
            scores, scores_pca, scores_ica, scores_kpca = my_svc_simple(subjects_tc_matrix, subjects_trial_label_matrix, target_name,my_cv_fold=20,my_comp=40)
            print("Classification Accuracy of SVC-RBF: {}, after decomposition using PCA:{}, ICA:{} and KPCA: {} \n"
                  .format(np.mean(scores),np.mean(scores_pca),np.mean(scores_ica),np.mean(scores_kpca)))

            scores, scores_pca, scores_ica, scores_kpca = my_svc_simple_subject_validation(subjects_tc_matrix,subjects_trial_label_matrix,
                                                                                           my_cv_fold=20,my_comp=40,my_testsize=0.25,my_valsize=0.25)
            print("Classification Accuracy of SVC-RBF: {}, after decomposition using PCA:{}, ICA:{} and KPCA: {} \n"
                  .format(np.mean(scores[-1]), np.mean(scores_pca[-1]), np.mean(scores_ica[-1]), np.mean(scores_kpca[-1])))

            test_acc, val_acc, train_acc = build_fc_nn_simple(subjects_tc_matrix, subjects_trial_label_matrix, target_name, layers=4, hidden_size=256)
            print("Classification Accuracy of Fully-connected neural networks: ")
            print("Accuracy of training {}, validation:{}, and testing: {} \n".format(train_acc,val_acc,test_acc))


    '''
    ####for script testing:
    pathfmri='/home/yuzhang/projects/rrg-pbellec/DATA/HCP/aws-s3_copy_022718/'
    pathout='/home/yuzhang/projects/rrg-pbellec/yuzhang/HCP/test/'
    mmp_atlas="/home/yuzhang/projects/rrg-pbellec/yuzhang/HCP/codes/HCP_S1200_GroupAvg_v1/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii"
    AtlasName='MMP'
    modality='MOTOR'
    
    python ./fmri_utils.py --fmri_folder=$pathfmri --temp_outputdir=$pathout --task_modality=$modality --atlas_name=$AtlasName --atlas_filename=$mmp_atlas --subject_to_start=0 --subject_to_last=10 --step_indicator=2 --subjectlist_index='t010'
    
    '''