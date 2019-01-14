#!/home/yuzhang/tensorflow-py3.6/bin/python3.6

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
from random import randint
from collections import defaultdict
#import matplotlib.pyplot as plt
###%matplotlib inline

from scipy import sparse
from nilearn import signal
from nilearn import image
from nilearn import connectome

from sklearn import linear_model
from sklearn import svm, metrics
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, train_test_split,ShuffleSplit
from sklearn.decomposition import PCA, FastICA, FactorAnalysis, DictionaryLearning, KernelPCA

try:
    from keras.utils import np_utils
    from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
    from keras.models import Model
    from keras.optimizers import SGD, Adam
except ImportError:
    print("Tensorflow is not avaliable in the current node!")
    print("deep learning models will not be running for this test!")


from tensorpack.utils import logger
from tensorpack import dataflow
from tensorpack.utils.utils import get_tqdm
from tensorpack.utils.serialize import dumps, loads
import lmdb
import model

##regression method to regress out head movement and other confounds
regr = linear_model.LinearRegression()


class hcp_task_fmri(object):
    """define task fmri parameters including modality, task-contrast, fmri-files and so on
        """

    def __init__(self, config):
        self.config = config
        self.modality = config.modality
        self.task_contrasts = config.task_contrasts
        self.Subject_Num = config.Subject_Num
        #self.Trial_Num = config.Trial_Num
        self.Node_Num = config.Node_Num
        self.TR = config.TR


    def bulid_dict_task_modularity(self):
        modality = self.modality
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
        mod_choices = {'mot': 'MOTOR',
                       'lan': 'LANGUAGE',
                       'emo': 'EMOTION',
                       'gam': 'GAMBLING',
                       'rel': 'RELATIONAL',
                       'soc': 'SOCIAL',
                       'wm': 'WM',
                       'all': 'ALLTasks'}
        task_choices = {'mot': motor_task_con,
                        'lan': lang_task_con,
                        'emo': emotion_task_con,
                        'gam': gambl_task_con,
                        'rel': reson_task_con,
                        'soc': social_task_con,
                        'wm': wm_task_con,
                        'all': all_task_con}

        self.modality = mod_choices.get(mod_chosen,'default')
        self.task_contrasts = task_choices.get(mod_chosen,'default')
        return self.task_contrasts, self.modality


    def load_fmri_data(self,postfix=None,confound_name=None):
        pathfmri = self.config.pathfmri
        startsub = self.config.startsub
        endsub = self.config.endsub
        self.subjectlist = self.config.subjectlist
        self.bulid_dict_task_modularity()
        modality = self.modality

        ###collect fmri time-series from fmri folders
        pathdata = Path(pathfmri)
        print("Collecting fmri data from folder: %s" % pathfmri)

        if not postfix:
            postfix = ".nii.gz"
        fmri_files = []
        confound_files = []
        subjects = []
        for fmri_file in sorted(pathdata.glob('tfMRI_' + modality + '*/*tfMRI_' + modality + '*_' + postfix)):
        #for fmri_file in sorted(pathdata.glob('*/FMRI/tfMRI_' + modality + '*/tfMRI_' + modality + '*_' + postfix)):
            subjects.append(Path(os.path.dirname(fmri_file)).parts[-3])
            fmri_files.append(str(fmri_file))

        if confound_name:
            for confound in sorted(pathdata.glob('tfMRI_' + modality + '*/*' + confound_name)):
            #for confound in sorted(pathdata.glob('*/FMRI/tfMRI_' + modality + '*/'+confound_name)):
                confound_files.append(str(confound))

        if not startsub:
            startsub = 0
            if not endsub:
                self.subjectlist = 'ALL'
        if (not endsub) or (endsub > len(np.unique(subjects))):
            endsub = len(np.unique(subjects))  ##len(fmri_files)

        ###change the list of fmri files
        sub_name = np.unique(subjects)[startsub:endsub]
        condition_mask = np.nonzero(pd.Series(subjects).isin(np.unique(sub_name)))[0]
        fmri_files = [fmri_files[ii] for ii in condition_mask]  # fmri_files[condition_mask]
        confound_files = [confound_files[ii] for ii in condition_mask]
        ##fmri_files = fmri_files[startsub:endsub]
        ##confound_files = confound_files[startsub:endsub]
        print('Collecting fmri files from subject {} to {}'.format(startsub, endsub))

        self.fmri_files = fmri_files
        self.confound_files = confound_files
        self.Subject_Num = len(fmri_files)
        tc_matrix = nib.load(fmri_files[0])
        self.Trial_Num, self.Node_Num = tc_matrix.shape   ##all fmri data from different subjects have the same length in time
        print("Data samples including %d subjects with %d trials and %d nodes for each" % (self.Subject_Num, self.Trial_Num, self.Node_Num))

        return self.fmri_files, self.confound_files


    def combine_events_spm(self):
        modality = self.modality
        task_contrasts = self.task_contrasts
        fmri_files = self.fmri_files

        ###combine the event design files into one csv file
        ###event design files are in the same folder of fmri files
        print("Different types of %s :" % modality)
        print(task_contrasts.keys())

        #Subject_Num = len(fmri_files)
        for subj in range(self.Subject_Num):
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


    def load_event_files(self,ev_filename=None):
        fmri_files = self.fmri_files
        confound_files = self.confound_files
        pathfmri = self.config.pathfmri
        pathout = self.config.pathout
        TR = self.config.TR
        Trial_Num = self.Trial_Num
        ##notes: all fmri data from different subjects have the same length in time, but event design files could be different for event-design
        Subject_Num = self.Subject_Num
        print('Num of trials: %d with TR=%2f \n' % (Trial_Num,TR))

        ###combining multiple sessions together
        pathdata = Path(pathfmri)
        '''
        EVS_files = []
        for ev,subcount in zip(sorted(pathdata.glob('*/FMRI/tfMRI_' + self.modality + '*/combined_events_spm_' + self.modality + '.csv')),np.arange(Subject_Num)):
            if subcount >= len(fmri_files):
                break
            if os.path.dirname(fmri_files[subcount]) == os.path.dirname(str(ev)):
                EVS_files.append(str(ev))
            else:
                print("Event files and fmri data are miss-matching for subject: ")
                print(Path(os.path.dirname(str(ev))).parts[-3::2], ':', Path(os.path.dirname(fmri_files[subcount])).parts[-3::2])
                print("Remove the residual fmri data for this subject: %s" % os.path.dirname(fmri_files[subcount]))
                del fmri_files[subcount]
                del confound_files[subcount]
        '''
        EVS_files = []
        subj = 0
        for ev, subcount in zip(sorted(pathdata.glob('tfMRI_' + self.modality + '*/*combined_events_spm_' + self.modality + '.csv')),np.arange(Subject_Num)):
            #for ev,subcount in zip(sorted(pathdata.glob('*/FMRI/tfMRI_' + self.modality + '*/combined_events_spm_' + self.modality + '.csv')),np.arange(Subject_Num)):
            ###remove fmri files if the event design is missing
            while os.path.dirname(fmri_files[subj]) < os.path.dirname(str(ev)):
                print("Event files and fmri data are miss-matching for subject: ")
                print(Path(os.path.dirname(str(ev))).parts[-3::2], ':', Path(os.path.dirname(fmri_files[subj])).parts[-3::2])
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

        self.EVS_files = EVS_files
        self.fmri_files = fmri_files
        self.confound_files = confound_files
        Subject_Num = min(Subject_Num, len(fmri_files))

        ###loading all event designs
        if not ev_filename:
            ev_filename = 'test.h5' #'.txt'

        events_all_subjects_file = pathout + self.modality + "_" + ev_filename
        if os.path.isfile(events_all_subjects_file):
            trial_infos = pd.read_csv(EVS_files[0],sep="\t",encoding="utf8",header = None,names=['onset','duration','rep','task'])
            Duras = np.ceil((trial_infos.duration/TR)).astype(int) #(trial_infos.duration/TR).astype(int)

            print('Collecting trial info from file:', events_all_subjects_file)
            '''
            subjects_trial_labels = pd.read_csv(events_all_subjects_file, sep="\t", encoding="utf8")
            print(subjects_trial_labels.keys())
    
            subjects_trial_label_matrix = subjects_trial_labels.loc[:, 'trial1':'trial' + str(Trial_Num)]
            sub_name = subjects_trial_labels['subject']
            coding_direct = subjects_trial_labels['coding']
            '''
            subjects_trial_label_matrix = pd.read_hdf(events_all_subjects_file, 'trials')
            subjects_trial_label_matrix = subjects_trial_label_matrix.values.tolist()
            sub_name = pd.read_hdf(events_all_subjects_file, 'subject')
            coding_direct = pd.read_hdf(events_all_subjects_file, 'coding')
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
                Onsets = list(np.around(trial_infos.onset / TR).astype(int))  # (trial_infos.onset/TR).astype(int)
                Duras = list(np.around(trial_infos.duration / TR).astype(int))  # (trial_infos.duration/TR).astype(int)
                Movetypes = trial_infos.task

                ##considering all subjects have the same length of fmri data
                TrialNum = Trial_Num
                '''
                #adjust the following script for event-design
                tc_matrix = nib.load(fmri_files[subj])
                TrialNum = tc_matrix.shape[0]
                if TrialNum > Trial_Num:
                    Trial_Num = TrialNum    ###subject-specific event design
                '''

                labels = ["rest"] * TrialNum
                for start, dur, move in zip(Onsets, Duras, Movetypes):
                    for ti in range(start - 1, start + dur):
                        #print(ti,TrialNum,start,dur)
                        if ti >= TrialNum:
                            continue
                        labels[ti] = self.task_contrasts[move]
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

        print("Data samples including %d event files consisting of %d subjects, each has %d coding directions and %d trials \n\n"
              % (len(sub_name), len(np.unique(sub_name)), np.ceil(len(coding_direct)/len(np.unique(sub_name))), np.array(subjects_trial_label_matrix).shape[1]))

        trial_dura = np.unique(Duras)[0]
        return subjects_trial_label_matrix,sub_name,coding_direct,trial_dura


    def load_event_files_tasktrials_only(self,ev_filename=None,window_size=None):
        fmri_files = self.fmri_files
        confound_files = self.confound_files
        pathfmri = self.config.pathfmri
        pathout = self.config.pathout
        TR = self.config.TR
        Trial_Num = self.Trial_Num
        Subject_Num = self.Subject_Num
        self.window_size = window_size

        ###combining multiple sessions together
        pathdata = Path(pathfmri)

        EVS_files = []
        subj = 0
        for ev,subcount in zip(sorted(pathdata.glob('*/FMRI/tfMRI_' + self.modality + '*/combined_events_spm_' + self.modality + '.csv')),np.arange(Subject_Num)):
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
        if len(EVS_files) != len(fmri_files):
            print('Miss-matching number of subjects between event:{} and fmri:{} files'.format(len(EVS_files), len(fmri_files)))

        self.EVS_files = EVS_files
        self.fmri_files = fmri_files
        self.confound_files = confound_files
        Subject_Num = min(Subject_Num,len(fmri_files))

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
        events_all_subjects_file = pathout + self.modality + window_str + "_" + ev_filename
        print(events_all_subjects_file)

        trial_infos = pd.read_csv(EVS_files[0], sep="\t", encoding="utf8", header=None,
                                  names=['onset', 'duration', 'rep', 'task'])
        Duras = np.ceil((trial_infos.duration / TR)).astype(int)
        self.Trial_Num = np.sum(np.floor(Duras / window_size_trial).astype(int))
        if os.path.isfile(events_all_subjects_file):
            trial_infos = pd.read_csv(EVS_files[0],sep="\t",encoding="utf8",header = None,names=['onset','duration','rep','task'])
            Duras = np.ceil((trial_infos.duration/TR)).astype(int) #(trial_infos.duration/TR).astype(int)

            print('Collecting trial info from file:', events_all_subjects_file)
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

                ##considering all subjects have the same length of fmri data
                TrialNum = Trial_Num
                '''
                #adjust the following script for event-design
                tc_matrix = nib.load(fmri_files[subj])
                TrialNum = tc_matrix.shape[0]
                if TrialNum > Trial_Num:
                    Trial_Num = TrialNum    ###subject-specific event design
                '''

                labels = ["rest"] * TrialNum
                start = 0
                for dur, move in zip(Duras, Movetypes):
                    dur_wind = np.floor(dur / window_size_trial).astype(int)
                    if dur_wind < 1:
                        dur_wind = 1
                    for ti in range(start, start + dur_wind):
                        if ti > TrialNum:
                            continue
                        labels[ti] = self.task_contrasts[move]
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

        subjects_trial_label_matrix = subjects_trial_label_matrix.values.tolist()
        print("Data samples including %d event files consisting of %d subjects, each has %d coding directions and %d trials"
              % (len(sub_name), len(np.unique(sub_name)), np.ceil(len(coding_direct)/len(np.unique(sub_name))),subjects_trial_label_matrix.shape[1]))

        trial_dura = np.unique(Duras)[0]
        return subjects_trial_label_matrix,sub_name,coding_direct,trial_dura


    #######################
    ###old function only for references
    def load_fmri_matrix_from_atlas(self, mmp_atlas, confound_files=None,fmri_matrix_name=None):
        ##loading fMRI signals using atlas mapping

        atlas_roi = nib.load(mmp_atlas).get_data()
        RegionLabels = [i for i in np.unique(atlas_roi) if i != 0]
        Region_Num = len(RegionLabels)
        print("%d regions in the parcellation map" % Region_Num)

        if not fmri_matrix_name:
            fmri_matrix_name = 'test.h5'  # 'test.txt'
        tcs_all_subjects_file = self.config.pathout + self.modality + "_" + fmri_matrix_name

        if os.path.isfile(tcs_all_subjects_file):
            ###loading time-series data from files
            '''
            subjects_tc_matrix = np.loadtxt(tcs_all_subjects_file).reshape((Subject_Num, Trial_Num, Region_Num))
            '''
            hdf = h5py.File(tcs_all_subjects_file, 'r')
            group1 = hdf.get('region_tc')
            group2 = hdf.get('sub_info')

            subjects_tc_matrix = []
            for subj in range(self.Subject_Num):
                n1 = np.array(group1.get('subj' + str(subj + 1)))
                subjects_tc_matrix.append(n1)
            sub_name = [sub.decode("utf-8") for sub in np.array(group2.get('sub_name'))]
            coding_direct = [sub.decode("utf-8") for sub in np.array(group2.get('coding'))]
            hdf.close()
            print(np.array(subjects_tc_matrix).shape)
        else:
            ###read fmri data through atlas mapping
            fmri_files = self.fmri_files
            if not confound_files:
                confound_files = [''] * len(fmri_files)
            else:
                confound_files = self.confound_files

            subjects_tc_matrix = []
            sub_name = []
            coding_direct = []
            for fmri_file, confound_file, sub_count in zip(fmri_files, confound_files, np.arange(0, Subject_Num)):
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
                ###using orthogonal matrix
                normed_matrix = preprocessing.normalize(confound, axis=0, norm='l2')
                confound, R = np.linalg.qr(normed_matrix)

                ##step2: regress out confound effects and save the residuals
                tc_matrix = preprocessing.scale(tc_matrix)
                regr.fit(confound, tc_matrix)
                mist_roi_tc = tc_matrix - np.matmul(confound, regr.coef_.T)

                subjects_tc_matrix.append(mist_roi_tc)
                if divmod(sub_count, 100)[1] == 0:
                    print("Processing subjects: %d" % sub_count)

            print(np.array(subjects_tc_matrix).shape)

            hdf5 = h5py.File(tcs_all_subjects_file, 'w')
            g1 = hdf5.create_group('region_tc')
            # g1.create_dataset('subjects',data=np.array(subjects_tc_matrix),compression="gzip")
            for subj in range(self.Subject_Num):
                g1.create_dataset('subj' + str(subj + 1), data=subjects_tc_matrix[subj], compression="gzip")
            g2 = hdf5.create_group('sub_info')
            g2.create_dataset('sub_name', data=np.array(sub_name).astype('|S9'), compression="gzip")
            g2.create_dataset('coding', data=np.array(coding_direct).astype('|S9'), compression="gzip")
            hdf5.close()  # closes the file

        return subjects_tc_matrix, sub_name, coding_direct

    def load_fmri_matrix_from_atlas_tasktrials_only(self, mmp_atlas, confound_files=None,fmri_matrix_name=None, window_size=None):
        ##loading fMRI signals within task trials using atlas mapping

        atlas_roi = nib.load(mmp_atlas).get_data()
        RegionLabels = [i for i in np.unique(atlas_roi) if i != 0]
        Region_Num = len(RegionLabels)
        print("%d regions in the parcellation map" % Region_Num)

        if not window_size:
            window_str = '_tasktrial'
            window_size_trial = 1
        else:
            window_size_trial = math.floor(window_size / TR)
            if window_size_trial < 1:
                window_size_trial = 1
            window_str = '_tasktrial_win' + str(window_size)
        if not fmri_matrix_name:
            fmri_matrix_name = 'test.h5'  # 'test.txt'
        tcs_all_subjects_file = self.config.pathout + self.modality + "_" + fmri_matrix_name

        if os.path.isfile(tcs_all_subjects_file):
            ###loading time-series data from files
            '''
            subjects_tc_matrix = np.loadtxt(tcs_all_subjects_file).reshape((Subject_Num, Trial_Num, Region_Num))
            '''
            hdf = h5py.File(tcs_all_subjects_file, 'r')
            group1 = hdf.get('region_tc')
            group2 = hdf.get('sub_info')

            subjects_tc_matrix = []
            for subj in range(self.Subject_Num):
                n1 = np.array(group1.get('subj' + str(subj + 1)))
                subjects_tc_matrix.append(n1)
            sub_name = [sub.decode("utf-8") for sub in np.array(group2.get('sub_name'))]
            coding_direct = [sub.decode("utf-8") for sub in np.array(group2.get('coding'))]
            hdf.close()
            print(np.array(subjects_tc_matrix).shape)
        else:
            ###read fmri data through atlas mapping
            Subject_Num = self.Subject_Num
            fmri_files = self.fmri_files
            EVS_files = self.EVS_files
            if not confound_files:
                confound_files = [''] * len(fmri_files)
            else:
                confound_files = self.confound_files

            subjects_tc_matrix = []
            sub_name = []
            coding_direct = []
            for fmri_file, confound_file, ev_file, sub_count in zip(fmri_files, confound_files, EVS_files,range(Subject_Num)):
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
                ##using orthogonal matrix
                normed_matrix = preprocessing.normalize(confound, axis=0, norm='l2')
                confound, R = np.linalg.qr(normed_matrix)

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

            hdf5 = h5py.File(tcs_all_subjects_file, 'w')
            g1 = hdf5.create_group('region_tc')
            # g1.create_dataset('subjects',data=np.array(subjects_tc_matrix),compression="gzip")
            for subj in range(Subject_Num):
                g1.create_dataset('subj' + str(subj + 1), data=subjects_tc_matrix[subj], compression="gzip")
            g2 = hdf5.create_group('sub_info')
            g2.create_dataset('sub_name', data=np.array(sub_name).astype('|S9'), compression="gzip")
            g2.create_dataset('coding', data=np.array(coding_direct).astype('|S9'), compression="gzip")
            hdf5.close()  # closes the file

        return subjects_tc_matrix, sub_name, coding_direct
    ####end of old fmri reading function
    #########################

    def prepare_fmri_files_list(self):

        ##pre-settings of the analysis
        task_contrasts, modality = self.bulid_dict_task_modularity()
        target_name = np.unique(list(task_contrasts.values()))
        print('Processing with the task_modality: %s with %d subtypes' % (self.modality, len(target_name)))
        print(target_name)

        ##collecting the fmri files
        fmri_filename = self.config.fmri_filename
        confound_filename = self.config.confound_filename
        fmri_files, confound_files = self.load_fmri_data(postfix=fmri_filename,confound_name=confound_filename)
        print('Generating Subject List of %s' % self.subjectlist)
        print('including %d fmri files and %d confounds files \n\n' % (len(fmri_files), len(confound_files)))

        ##collecting the event design files and loading fmri data
        #self.combine_events_spm()

        print('Collecting event design files for subjects and saved into matrix ...')
        self.ev_filename = 'event_labels_1200R' + '_test_Dec2018_' + self.subjectlist + '.h5'  # '.txt'
        self.fmri_matrix_filename = self.config.AtlasName + '_ROI_act_1200R' + '_test_Dec2018_' + self.subjectlist + '.lmdb'  # '.h5' # '.txt'
        self.lmdb_filename = self.config.pathout + self.modality + '_' + self.fmri_matrix_filename
        subjects_trial_label_matrix, sub_name, coding,trial_dura = self.load_event_files(ev_filename=self.ev_filename)

        return subjects_trial_label_matrix, sub_name, coding,trial_dura


####end of hcp_task_fmri class
#####################################

class hcp_rsfmri(hcp_task_fmri):
    """define task fmri parameters including modality, task-contrast, fmri-files and so on
        """
    def load_fmri_data(self, sub_name=None,postfix=None,confound_name=None):
        pathfmri = self.config.pathfmri
        startsub = self.config.startsub
        endsub = self.config.endsub
        self.subjectlist = self.config.subjectlist
        self.sub_name = None
        self.session_id = None
        self.modality = 'REST'

        ###collect fmri time-series from fmri folders
        pathdata = Path(pathfmri)
        print("Collecting rs-fmri data from folder: %s" % pathfmri)

        if not postfix:
            postfix = "Atlas_hp2000_clean.dtseries.nii" #".nii.gz"
        fmri_files = []
        confound_files = []
        WM_files = []
        CSF_files = []
        subjects = []
        for fmri_file in sorted(pathdata.glob('*/FMRI/rfMRI_' + self.modality + '*/rfMRI_' + self.modality + '*_' + postfix)):
            subjects.append(Path(os.path.dirname(fmri_file)).parts[-3])
            fmri_files.append(str(fmri_file))

        if confound_name:
            for confound in sorted(pathdata.glob('*/FMRI/rfMRI_' + self.modality + '*/'+confound_name)):
                confound_files.append(str(confound))

        for file in sorted(pathdata.glob('*/FMRI/rfMRI_' + self.modality + '*/rfMRI_' + self.modality +'*_WM.txt')):
            WM_files.append(str(file))

        for file in sorted(pathdata.glob('*/FMRI/rfMRI_' + self.modality + '*/rfMRI_' + self.modality +'*_CSF.txt')):
            CSF_files.append(str(file))

        if not startsub:
            startsub = 0
            endsub = int(round(self.config.Subject_Num))
            self.subjectlist = 'ALL'
        if (not endsub) or (endsub > len(np.unique(subjects))):
            endsub = len(np.unique(subjects))

        ###change the list of fmri files
        print(len(fmri_files), len(confound_files), len(WM_files), len(CSF_files))
        if not sub_name:
            sub_name = np.unique(subjects)[startsub:endsub]
            self.task_corresponding = False
            print('Collecting rs-fmri files from subject {} to {}'.format(startsub, endsub))
        else:
            self.task_corresponding = True
            print('Collecting rs-fmri files according to {} task selections of {} subjects'.format(self.config.modality,len(subjects)))
        condition_mask = np.nonzero(pd.Series(subjects).isin(np.unique(sub_name)))[0]
        fmri_files = [fmri_files[ii] for ii in condition_mask]  # fmri_files[condition_mask]
        confound_files = [confound_files[ii] for ii in condition_mask]  # confound_files[condition_mask]
        WM_files = [WM_files[ii] for ii in condition_mask]  # WM_files[condition_mask]
        CSF_files = [CSF_files[ii] for ii in condition_mask]  # CSF_files[condition_mask]
        subjects = [subjects[ii] for ii in condition_mask]

        self.fmri_files = fmri_files
        self.confound_files = confound_files
        self.WM_files = WM_files
        self.CSF_files = CSF_files
        self.Subject_Num = len(fmri_files)
        self.sub_name = subjects


        tc_matrix = nib.load(fmri_files[0])
        self.Trial_Num, self.Node_Num = tc_matrix.shape   ##all fmri data from different subjects have the same length in time
        print("Data samples including %d subjects with %d nodes for each" % (self.Subject_Num, self.Node_Num))

        return self.fmri_files, self.confound_files, self.WM_files, self.CSF_files, self.sub_name

    ###################
    ###old function only for references
    def load_rsfmri_matrix_from_atlas(self, mmp_atlas, confound_filename=None, wm_flag=True, csf_flag=True, fmri_matrix_name=None):
        ##loading fMRI signals using atlas mapping

        atlas_roi = nib.load(mmp_atlas).get_data()
        RegionLabels = [i for i in np.unique(atlas_roi) if i != 0]
        Region_Num = len(RegionLabels)
        print("%d regions in the parcellation map" % Region_Num)

        if not fmri_matrix_name:
            fmri_matrix_name = 'test.h5'  # 'test.txt'
        tcs_all_subjects_file = self.config.pathout + self.modality + "_" + fmri_matrix_name

        if os.path.isfile(tcs_all_subjects_file):
            ###loading time-series data from files
            '''
            subjects_tc_matrix = np.loadtxt(tcs_all_subjects_file).reshape((Subject_Num, Trial_Num, Region_Num))
            '''
            hdf = h5py.File(tcs_all_subjects_file, 'r')
            group1 = hdf.get('region_tc')
            group2 = hdf.get('sub_info')

            subjects_tc_matrix = []
            for subj in range(self.Subject_Num):
                n1 = np.array(group1.get('subj' + str(subj + 1)))
                subjects_tc_matrix.append(n1)
            sub_name = [sub.decode("utf-8") for sub in np.array(group2.get('sub_name'))]
            session_id = [sub.decode("utf-8") for sub in np.array(group2.get('session'))]
            coding_direct = [sub.decode("utf-8") for sub in np.array(group2.get('coding'))]
            hdf.close()
            print(np.array(subjects_tc_matrix).shape)
        else:

            ###read fmri data through atlas mapping
            Subject_Num = self.Subject_Num
            fmri_files = self.fmri_files
            if not confound_filename:
                confound_files = [''] * len(fmri_files)
            else:
                confound_files = self.confound_files
            if not wm_flag:
                WM_files = [''] * len(fmri_files)
            else:
                WM_files = self.WM_files
            if not csf_flag:
                CSF_files = [''] * len(fmri_files)
            else:
                CSF_files = self.CSF_files

            subjects_tc_matrix = []
            sub_name = []
            session_id = []
            coding_direct = []
            for fmri_file, confound_file, wm_file, csf_file, sub_count in zip(fmri_files, confound_files,WM_files,CSF_files,range(Subject_Num)):
                pathsub = Path(os.path.dirname(fmri_files[sub_count]))
                if os.path.dirname(fmri_file) != os.path.dirname(str(confound_file)):
                    print('Mis-matching subject between fmri data %s and confounding files %s ' %
                          (Path(os.path.dirname(fmri_file)).parts[-3], Path(os.path.dirname(confound_file)).parts[-3]))
                if os.path.dirname(fmri_file) != os.path.dirname(str(wm_file)):
                    print('Mis-matching subject between fmri data %s and confounding files %s ' %
                          (Path(os.path.dirname(fmri_file)).parts[-3], Path(os.path.dirname(confound_file)).parts[-3]))
                if os.path.dirname(fmri_file) != os.path.dirname(str(csf_file)):
                    print('Mis-matching subject between fmri data %s and confounding files %s ' %
                          (Path(os.path.dirname(fmri_file)).parts[-3], Path(os.path.dirname(confound_file)).parts[-3]))
                sub_name.append(pathsub.parts[-3])
                session_id.append(pathsub.parts[-1].split('_')[-2][-1])
                coding_direct.append(pathsub.parts[-1].split('_')[-1])

                ##step1:load fmri time-series and confounds into matrix
                print('Read data from: ', fmri_file)
                tc_roi_matrix = map_surface_to_parcel_fast(atlas_roi, fmri_file)
                if tc_roi_matrix is None:
                    print("Skip data loading for subject:", fmri_file)
                    del sub_name[-1]
                    del session_id[-1]
                    del coding_direct[-1]
                    break
                tc_matrix = np.transpose(np.array(tc_roi_matrix), (1, 0))
                if confound_file == '':
                    confound_move = np.ones((tc_matrix.shape[0],), dtype=int)
                else:
                    confound_move = np.loadtxt(confound_file)
                if wm_file == '':
                    confound_wm = np.ones((tc_matrix.shape[0],), dtype=int)
                else:
                    confound_wm = np.expand_dims(np.loadtxt(wm_file),1)
                if csf_file == '':
                    confound_csf = np.ones((tc_matrix.shape[0],), dtype=int)
                else:
                    confound_csf = np.expand_dims(np.loadtxt(csf_file),1)
                ###using orthogonal matrix
                confound_matrix = np.hstack((confound_move, confound_wm, confound_csf))
                normed_matrix = preprocessing.normalize(confound_matrix, axis=0, norm='l2')
                confound_matrix, R = np.linalg.qr(normed_matrix)

                ###step2: regress out confound effects and save the residuals
                from nilearn import signal
                tc_matrix_clean = signal.clean(tc_matrix, detrend=False, standardize=True, confounds=confound_matrix,
                                               low_pass=self.config.highcut, high_pass=self.config.lowcut, t_r=self.TR)
                subjects_tc_matrix.append((tc_matrix_clean)) ##np.transpose
                if divmod(sub_count, 100)[1] == 0:
                    print("Processing subjects: %d \n\n" % sub_count)

            print(np.array(subjects_tc_matrix).shape)

            hdf5 = h5py.File(tcs_all_subjects_file, 'w')
            g1 = hdf5.create_group('region_tc')
            # g1.create_dataset('subjects',data=np.array(subjects_tc_matrix),compression="gzip")
            for subj in range(Subject_Num):
                g1.create_dataset('subj' + str(subj + 1), data=subjects_tc_matrix[subj], compression="gzip")
            g2 = hdf5.create_group('sub_info')
            g2.create_dataset('sub_name', data=np.array(sub_name).astype('|S9'), compression="gzip")
            g2.create_dataset('session', data=np.array(session_id).astype('|S9'), compression="gzip")
            g2.create_dataset('coding', data=np.array(coding_direct).astype('|S9'), compression="gzip")
            hdf5.close()  # closes the file

        self.sub_name = sub_name
        self.session_id = session_id
        self.coding_direct = coding_direct

        return subjects_tc_matrix, sub_name, session_id, coding_direct

    def cal_functional_connectome_matrices(self,subjects_tc_matrix,corr_kind='correlation',corr_mat_name=None):
        ##calculate functional connectivity between regions
        Subject_Num = np.array(subjects_tc_matrix).shape[0]
        TIME, Region_Num = subjects_tc_matrix[0].shape
        print('Calculating functional connectivity matrix between {} Regions for {} Subjects'.format(Region_Num,Subject_Num))

        if not corr_mat_name:
            corr_mat_name = 'test.h5'  # 'test.txt'
        corr_all_subjects_file = self.config.pathout + self.modality + "_" + corr_mat_name

        if self.session_id is None:
            sub_name = []
            session_id = []
            coding_direct = []
            for fmri_file in self.fmri_files:
                pathsub = Path(os.path.dirname(fmri_file))
                sub_name.append(pathsub.parts[-3])
                session_id.append(pathsub.parts[-1].split('_')[-2][-1])
                coding_direct.append(pathsub.parts[-1].split('_')[-1])
            self.sub_name = sub_name
            self.session_id = session_id
            self.coding_direct = coding_direct

        if os.path.isfile(corr_all_subjects_file):
            ###loading RSFC data from files
            '''
            subjects_tc_matrix = np.loadtxt(tcs_all_subjects_file).reshape((Subject_Num, Trial_Num, Region_Num))
            '''
            hdf = h5py.File(corr_all_subjects_file, 'r')
            group1 = hdf.get('FC_corr_mat')
            group2 = hdf.get('FC_corr_mat_z')
            group3 = hdf.get('sub_info')

            mean_corr_matrix = np.array(group1.get('mean_corr_mat'))
            mean_corr_matrix_z = np.array(group2.get('mean_corr_mat_z'))
            corr_matrix = []
            corr_matrix_z = []
            for subj in range(self.Subject_Num):
                n1 = np.array(group1.get('subj' + str(subj + 1)))
                corr_matrix.append(n1)

                n2 = np.array(group2.get('subj' + str(subj + 1)))
                corr_matrix_z.append(n2)
            sub_name = [sub.decode("utf-8") for sub in np.array(group3.get('sub_name'))]
            session_id = [sub.decode("utf-8") for sub in np.array(group3.get('session'))]
            coding_direct = [sub.decode("utf-8") for sub in np.array(group3.get('coding'))]
            hdf.close()
            print(np.array(corr_matrix).shape)
        else:

            print('using %s for connectivity measure...' % corr_kind)
            connectome_measure = connectome.ConnectivityMeasure(kind=corr_kind)
            corr_matrix = connectome_measure.fit_transform(subjects_tc_matrix)

            ##fisher z-transform
            corr_matrix[corr_matrix >= 1.0] = 0.9999   ##to avoid dividing by zero
            corr_matrix[corr_matrix <= -1.0] = -0.9999
            corr_matrix_z = np.arctanh(corr_matrix)
            mean_corr_matrix_z = np.mean(corr_matrix_z, axis=0)
            #print(mean_corr_matrix_z.shape)
            #plt.matshow(mean_corr_matrix_z, cmap=plt.cm.BuPu_r)
            #plt.colorbar()

            mean_corr_matrix = np.tanh(mean_corr_matrix_z)
            #plt.matshow(mean_corr_matrix, cmap=plt.cm.BuPu_r)
            #plt.colorbar()

            print(corr_matrix.shape)
            hdf5 = h5py.File(corr_all_subjects_file, 'w')

            g1 = hdf5.create_group('FC_corr_mat')
            g1.create_dataset('mean_corr_mat', data=np.array(mean_corr_matrix), compression="gzip")
            for subj in range(Subject_Num):
                g1.create_dataset('subj' + str(subj + 1), data=corr_matrix[subj], compression="gzip")

            g2 = hdf5.create_group('FC_corr_mat_z')
            g2.create_dataset('mean_corr_mat_z', data=np.array(mean_corr_matrix_z), compression="gzip")
            for subj in range(Subject_Num):
                g2.create_dataset('subj' + str(subj + 1), data=corr_matrix_z[subj], compression="gzip")
            g3 = hdf5.create_group('sub_info')
            g3.create_dataset('sub_name', data=np.array(self.sub_name).astype('|S9'), compression="gzip")
            g3.create_dataset('session', data=np.array(self.session_id).astype('|S9'), compression="gzip")
            g3.create_dataset('coding', data=np.array(self.coding_direct).astype('|S9'), compression="gzip")
            hdf5.close()  # closes the file

        return mean_corr_matrix, corr_matrix, corr_matrix_z


    def prepare_rsfmri_files_list(self,sub_name=None, N_thread=None):

        ##collecting the fmri files
        rsfmri_filename = self.config.rsfmri_filename
        confound_filename = self.config.confound_filename
        fmri_files, confound_files, wm_files, csf_files, sub_name = self.load_fmri_data(sub_name=sub_name,postfix=rsfmri_filename,confound_name=confound_filename)
        print('Generating Subject List of %s' % self.subjectlist)
        print("including {} fmri files and confounds files for movement:{}, wm:{} and csf {} \n\n".
              format(len(fmri_files), len(confound_files),len(wm_files),len(csf_files)))

        ##collecting rs-fmri time-series
        mmp_atlas = self.config.mmp_atlas
        confound_filename = self.config.confound_filename
        if self.task_corresponding:
            fmri_matrix_filename = self.config.AtlasName + '_ROI_act_1200R' + '_task_' + self.config.modality + '.h5'
        else:
            fmri_matrix_filename = self.config.AtlasName + '_ROI_act_1200R' + '_' + self.subjectlist + '.h5'
        ##subjects_tc_matrix, sub_name, session_id, coding_direct = self.load_rsfmri_matrix_from_atlas(mmp_atlas, confound_filename=confound_filename, fmri_matrix_name=fmri_matrix_filename)

        lmdb_filename = self.config.pathout + self.modality + '_' + fmri_matrix_filename.split('.')[0] + '.lmdb'
        if os.path.isfile(lmdb_filename):
            lmdb_filename = lmdb_filename.split('.')[0] + '2.lmdb'
        print('Saving rs-fmri time-courses into tmp-file: %s' % lmdb_filename)

        if not N_thread:
            N_thread = self.config.n_thread
        confound_files_merge = pd.DataFrame(list(zip(confound_files, wm_files, csf_files)),
                                            columns=['headmove','wm', 'csf'])
        subjects_tc_matrix, subname_coding = extract_mean_seris_thread(fmri_files, confound_files_merge, mmp_atlas,lmdb_filename,self.Trial_Num,
                                                                       nr_thread=N_thread,
                                                                       buffer_size=self.config.n_buffersize)

        ##calculating functional connectome
        if self.task_corresponding:
            corr_mat_filename = self.config.AtlasName + '_rsfc_matrix_1200R' + '_task_' + self.config.modality + '.h5'
        else:
            corr_mat_filename = self.config.AtlasName + '_rsfc_matrix_1200R' + '_' + self.subjectlist + '.h5'
        if os.path.isfile(corr_mat_filename):
            corr_mat_filename = corr_mat_filename.split('.')[0] + '2.h5'
        print('Saving rsfc-matrix into tmp-file: %s' % corr_mat_filename)

        mean_corr_matrix, corr_matrix, corr_matrix_z = self.cal_functional_connectome_matrices(subjects_tc_matrix,corr_kind='correlation',corr_mat_name=corr_mat_filename)
        print(mean_corr_matrix.shape)

        return subjects_tc_matrix, mean_corr_matrix
####end of hcp_rsfmri class
#####################################


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
        if isinstance(self.confound_files, pd.DataFrame):
            for a, b in zip(self.fmri_files, self.confound_files.values):
                yield a, b
        else:
            for a, b in zip(self.fmri_files, self.confound_files):
                yield a, b


def map_surface_to_parcel(atlas_roi, fmri_file):

    #atlas_roi = nib.load(parcel_map).get_data()
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


def map_surface_to_parcel_fast(atlas_roi, fmri_file):
    ##build a dataframe to extract mean time-series for each roi within each trial

    # atlas_roi = nib.load(parcel_map).get_data()
    RegionLabels = [i for i in np.unique(atlas_roi) if i != 0]
    Region_Num = len(RegionLabels)

    try:
        tc = nib.load(str(fmri_file))
    except:
        print("Corrupted nifti file: ", str(fmri_file))
        return None

    tc_matrix = nib.load(str(fmri_file)).get_data()
    if atlas_roi.shape[1] != tc_matrix.shape[1]:
        print("Dimension of parcellation map not matching")
        print("%d vs %d " % (atlas_roi.shape[1], tc_matrix.shape[1]))
        tc_matrix = tc_matrix[:, range(atlas_roi.shape[1])]

    print('Read data from: ', fmri_file)
    Trial_Num, Node_Num = tc_matrix.shape
    tc_matrix_df = pd.DataFrame(data=tc_matrix.ravel(), columns=['tc_signal'])
    tc_matrix_df['roi_label'] = np.repeat(atlas_roi, Trial_Num, axis=0).ravel()
    tc_matrix_df['trial_id'] = np.repeat(np.arange(Trial_Num).reshape((Trial_Num, 1)), Node_Num, axis=1).ravel()
    # df = pd.DataFrame(values, index=index)

    tc_roi = tc_matrix_df.groupby(['roi_label', 'trial_id'], as_index=False).mean()
    tc_roi_matrix = tc_roi['tc_signal'][tc_roi['roi_label'] != 0]
    tc_roi_matrix = tc_roi_matrix.values.reshape(Region_Num, Trial_Num)

    print('Done reading fmri data ', fmri_file)
    # print(np.transpose(np.array(tc_roi_matrix),(1,0)).shape)
    return tc_roi_matrix


def map_func_extract_seris(dp, mmp_atlas_roi):
    fmri_file = dp[0]
    confound_file = dp[1]

    #tc_roi_matrix = map_surface_to_parcel(mmp_atlas_roi, fmri_file)
    tc_roi_matrix = map_surface_to_parcel_fast(mmp_atlas_roi, fmri_file)
    if tc_roi_matrix is None:
        print("Skip data loading for subject: ", fmri_file)
        return [fmri_file, None]

    tc_matrix = np.transpose(np.array(tc_roi_matrix), (1, 0))
    mist_roi_tc = preprocessing.scale(tc_matrix)
    '''
    print('Regress out head motion from: ',confound_file)
    ##regress out confound effects
    confound = np.loadtxt(confound_file)
    ###using orthogonal matrix
    normed_matrix = preprocessing.normalize(confound, axis=0, norm='l2')
    confound, R = np.linalg.qr(normed_matrix)
    mist_roi_tc = signal.clean(tc_matrix, detrend=False, standardize=True, confounds=confound)
    
    tc_matrix = preprocessing.scale(tc_matrix)
    regr.fit(confound, tc_matrix)
    mist_roi_tc = tc_matrix - np.matmul(confound, regr.coef_.T)
    '''

    print('Done regression of confounds ')
    return [fmri_file, mist_roi_tc]

def map_func_extract_resting(dp, mmp_atlas_roi,lowcut=0.01,highcut=0.08,TR=0.72):
    fmri_file = dp[0]
    confound_file = dp[1]
    headmove_file = confound_file[0]
    wm_file = confound_file[1]
    csf_file = confound_file[2]
    #print(len(confound_file),headmove_file,wm_file,csf_file)

    tc_roi_matrix = map_surface_to_parcel_fast(mmp_atlas_roi, fmri_file)
    if tc_roi_matrix is None:
        print("Skip data loading for subject:", fmri_file)
        return [fmri_file, None]
    print('Regress out head motion from: ',confound_file)
    tc_matrix = np.transpose(np.array(tc_roi_matrix), (1, 0))

    confound_move = np.loadtxt(headmove_file)
    confound_wm = np.expand_dims(preprocessing.scale(np.loadtxt(wm_file)),1)
    confound_csf = np.expand_dims(preprocessing.scale(np.loadtxt(csf_file)),1)
    confound_matrix = np.hstack((confound_move, confound_wm, confound_csf))
    ###using orthogonal matrix
    normed_matrix = preprocessing.normalize(confound_matrix, axis=0, norm='l2')
    confound_matrix, R = np.linalg.qr(normed_matrix)

    ###step2: regress out confound effects and save the residuals
    tc_matrix_clean = signal.clean(tc_matrix, detrend=False, standardize=True, confounds=confound_matrix,
                                   low_pass=highcut, high_pass=lowcut, t_r=TR)

    print('Done regression of confounds ')
    return [fmri_file, tc_matrix_clean]


def dump_mean_seris_to_lmdb(df, lmdb_path, write_frequency=10):
    """
    save extracted series into lmdb:lmdb_path
    """
    assert isinstance(df, dataflow.DataFlow), type(df)
    isdir = os.path.isdir(lmdb_path)
    df.reset_state()
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=int(1e12) * 2, readonly=False,
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
            keystr = u'{}'.format(dp[0]).encode('ascii')
            txn.put(keystr, dumps(dp[1].astype('float32')))  ##change datatype to float32
            #keys += [keystr]
            pbar.update()
            print(idx, keystr)

            if (idx + 1) % write_frequency == 0:
                txn.commit()
                txn = db.begin(write=True)
                # print(idx+1)

        # write last batch
        txn.commit()

        #with db.begin(write=True) as txn:
        #    txn.put(b'__keys__', dumps(keys))

        logger.info("Flushing database ...")
        db.sync()

    db.close()
    logger.info("Closing LMDB files ...")


def extract_mean_seris_thread(fmri_files, confound_files, mmp_atlas, lmdb_filename, Trial_Num,nr_thread=100, buffer_size=50):
    ####extract roi mean time series and save to lmdb file

    buffer_size = min(len(fmri_files),buffer_size)
    nr_thread = min(len(fmri_files),nr_thread)

    if not os.path.isfile(lmdb_filename):
        mmp_atlas_roi = nib.load(mmp_atlas).get_data()
        ds0 = dataflow_fmri_with_confound(fmri_files, confound_files)
        print('dataflowSize is ' + str(ds0.size()))

        #print('buffer_size is ' + str(buffer_size))
        print('Loading data using %d threads with %d buffer_size ... \n\n' % (nr_thread, buffer_size))

        ####running the model
        start_time = time.clock()
        if any('REST' in string for string in lmdb_filename.split('_')):
            ds1 = dataflow.MultiThreadMapData(
                ds0, nr_thread=nr_thread,
                map_func=lambda dp: map_func_extract_resting(dp, mmp_atlas_roi,lowcut=0.01,highcut=0.08,TR=0.72),
                buffer_size=buffer_size,
                strict=True)
        else:
            ds1 = dataflow.MultiThreadMapData(
                ds0, nr_thread=nr_thread,
                map_func=lambda dp: map_func_extract_seris(dp, mmp_atlas_roi),
                buffer_size=buffer_size,
                strict=True)
        ds1 = dataflow.PrefetchDataZMQ(ds1, nr_proc=1)
        ##ds1._reset_once()
        ds1.reset_state()

        dump_mean_seris_to_lmdb(ds1, lmdb_filename, write_frequency=buffer_size)
        print('Time Usage of loading data in seconds: {}'.format(time.clock() - start_time))

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
            #print(key)
            if key == b'__keys__':
                continue
            pathsub = Path(os.path.dirname(key.decode("utf-8")))
            if any('REST' in string for string in lmdb_filename.split('_')):
                fmri_sub_name.append(pathsub.parts[-3] + '_' + pathsub.parts[-1].split('_')[-2][-1] + '_' + pathsub.parts[-1].split('_')[-1])
            else:
                fmri_sub_name.append(pathsub.parts[-3] + '_' + pathsub.parts[-1].split('_')[-1])
            data = loads(lmdb_txn.get(key))
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

    # print(np.array(matrix_dict).shape)
    # print(fmri_sub_name)
    return matrix_dict, fmri_sub_name


def extract_mean_seris(fmri_files, confound_files, mmp_atlas, lmdb_filename, Trial_Num, nr_proc=100, buffer_size=50):
    ####extract roi mean time series and save to lmdb file

    buffer_size = min(len(fmri_files),buffer_size)
    nr_proc = min(len(fmri_files),nr_proc)


    if not os.path.isfile(lmdb_filename):
        mmp_atlas_roi = nib.load(mmp_atlas).get_data()
        ds0 = dataflow_fmri_with_confound(fmri_files, confound_files)
        print('dataflowSize is ' + str(ds0.size()))

        print('buffer_size is ' + str(buffer_size))

        ####running the model
        start_time = time.clock()
        ds1 = dataflow.MultiProcessMapDataZMQ(
            ds0, nr_proc=nr_proc,
            map_func=lambda dp: map_func_extract_seris(dp, mmp_atlas_roi),
            buffer_size=buffer_size,
            strict=True)
        ds1._reset_once()

        dump_mean_seris_to_lmdb(ds1, lmdb_filename, write_frequency=buffer_size)
        print('Time Usage of loading data in seconds: {}'.format(time.clock() - start_time))

    ## read lmdb matrix
    matrix_dict = []
    fmri_sub_name = []
    lmdb_env = lmdb.open(lmdb_filename, subdir=False)
    try:
        lmdb_txn = lmdb_env.begin()
        listed_fmri_files = loads(lmdb_txn.get(b'__keys__'))
        listed_fmri_files = [l.decode("utf-8") for l in listed_fmri_files]
        print('Stored fmri data from files:')
        # print(listed_fmri_files)
    except:
        print('Search each key for every fmri file...')

    with lmdb_env.begin() as lmdb_txn:
        cursor = lmdb_txn.cursor()
        for key, value in cursor:
            #print(key)
            if key == b'__keys__':
                continue
            pathsub = Path(os.path.dirname(key.decode("utf-8")))
            fmri_sub_name.append(pathsub.parts[-3] + '_' + pathsub.parts[-1].split('_')[-1])
            data = loads(lmdb_txn.get(key))
            if data.shape[0] != Trial_Num:
                print('fmri data shape mis-matching between subjects...')
                print('Check subject:  %s with only %d Trials \n' % (fmri_sub_name[-1],data.shape[0]))
            matrix_dict.append(np.array(data))
    lmdb_env.close()

    print(np.array(matrix_dict).shape)
    #print(fmri_sub_name)
    return matrix_dict, fmri_sub_name

###end of tensorpack: multithread
##############################################################


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


def preclean_data_for_shape_match(subjects_tc_matrix,subjects_trial_label_matrix, fmri_sub_name):
    print("Pre-clean the fmri and event data to make sure the matching shapes between two arrays!")
    Subject_Num = np.array(subjects_tc_matrix).shape[0]
    Trial_Num, Region_Num = subjects_tc_matrix[0].shape
    if np.array(subjects_trial_label_matrix).shape[0] != Subject_Num:
        print('Warning: Mis-matching subjects list between fmri-data-matrix and trial-label-matrix')
        print(np.array(subjects_tc_matrix).shape,np.array(subjects_trial_label_matrix).shape)

    for subj in range(Subject_Num):
        try:
            tsize, rsize = subjects_tc_matrix[subj].shape
        except:
            print(subj==Subject_Num-1)
            print('The end of SubjectList...\n')
        if tsize != Trial_Num:
            print('Remove subject: %s due to different trial num: %d in the fmri data' % (fmri_sub_name[subj],tsize))
            del subjects_tc_matrix[subj]
            del subjects_trial_label_matrix[subj]
        if rsize != Region_Num:
            print('Remove subject: %s due to different region num: %d in the fmri data' % (fmri_sub_name[subj],rsize))
            del subjects_tc_matrix[subj]
            del subjects_trial_label_matrix[subj]

    print('Done matching data shapes:',np.array(subjects_tc_matrix).shape,np.array(subjects_trial_label_matrix).shape)
    return subjects_tc_matrix, subjects_trial_label_matrix


def train_test_split_dataframe_simple(tc_matrix, label_matrix, sub_num=None, n_folds=20, testsize=0.2, valsize=0.2, randomseed=1234):
    Subject_Num, Trial_Num, Region_Num = np.array(tc_matrix).shape
    rs = np.random.RandomState(randomseed)
    if not sub_num or sub_num>Subject_Num:
        sub_num = Subject_Num
    tc_matrix = np.array(tc_matrix[:sub_num])
    label_matrix = np.array(label_matrix[:sub_num]).reshape(sub_num * Trial_Num, )

    train_sid_tmp, test_sid = train_test_split(range(sub_num), test_size=testsize, random_state=rs, shuffle=True)

    ###build a dataframe to support subject selection
    label_matrix_df = pd.DataFrame(data=np.array(tc_matrix).ravel(), columns=['tc_signal'])
    label_matrix_df['target_label'] = np.tile(label_matrix, (1, Region_Num)).ravel()
    label_matrix_df['subj_id'] = np.tile(np.arange(Subject_Num), (1, Trial_Num, Region_Num)).ravel()
    label_matrix_df['trial_id'] = np.tile(np.arange(Trial_Num), (Subject_Num, 1, Region_Num)).ravel()
    label_matrix_df2 = label_matrix_df[label_matrix_df['target_label'] != 'rest']

    ##split dataset into train and testing
    fmri_data = label_matrix_df2['tc_signal'].reshape(-1, Region_Num)
    scaler = preprocessing.StandardScaler().fit(fmri_data)
    fmri_data_test = label_matrix_df2['tc_signal'][label_matrix_df2['subj_id'].isin(test_sid)]
    label_data_test = label_matrix_df2['target_label'][label_matrix_df2['subj_id'].isin(test_sid)]
    fmri_data_test = fmri_data_test.reshape(-1, Region_Num)
    label_data_test = label_data_test.reshape(-1, Region_Num)[:, 0]
    X_test = scaler.transform(fmri_data_test)
    Y_test = label_data_test
    print(fmri_data_test.shape, fmri_data_test.shape)

    from sklearn.model_selection import ShuffleSplit
    valsplit = ShuffleSplit(n_splits=10, test_size=0.1, random_state=10)
    X_train_scaled = []
    X_val_scaled = []
    Y_train_scaled = []
    Y_val_scaled = []
    from heapq import merge
    for train_sid, val_sid in valsplit.split(train_sid_tmp):
        fmri_data_train = label_matrix_df2['tc_signal'][label_matrix_df2['subj_id'].isin(train_sid)]
        label_data_train = label_matrix_df2['target_label'][label_matrix_df2['subj_id'].isin(train_sid)]
        print(fmri_data_train.shape, label_data_train.shape)
        X_train_scaled.append(scaler.transform(fmri_data_train.reshape(-1, Region_Num)))
        Y_train_scaled.append(label_data_train.reshape(-1, Region_Num)[:, 0])

        fmri_data_val = label_matrix_df2['tc_signal'][label_matrix_df2['subj_id'].isin(val_sid)]
        label_data_val = label_matrix_df2['target_label'][label_matrix_df2['subj_id'].isin(val_sid)]
        X_val_scaled.append(scaler.transform(fmri_data_val.reshape(-1, Region_Num)))
        Y_val_scaled.append(label_data_val.reshape(-1, Region_Num)[:, 0])

    return X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled, X_test, Y_test


def my_svc_simple(subjects_tc_matrix,subjects_trial_label_matrix,target_name,sub_num=None, block_dura=18, Flag_PCA=1, my_cv_fold=10,my_comp=20):
    ###using svm for classification with each trial/time as one sample

    Subject_Num, Trial_Num, Region_Num = np.array(subjects_tc_matrix).shape
    if Trial_Num != np.array(subjects_trial_label_matrix).shape[1]:
        print('Miss-matching trial infos for event and fmri data')
    if Subject_Num != np.array(subjects_trial_label_matrix).shape[0]:
        print('Adjust subject numbers for event data')
        subjects_trial_label_matrix = np.array(subjects_trial_label_matrix[:Subject_Num])
    else:
        subjects_trial_label_matrix = np.array(subjects_trial_label_matrix)

    ###SVM classifier
    ##feature matrix
    if not sub_num or sub_num>Subject_Num:
        sub_num = Subject_Num
    if not block_dura:
        block_dura = 18 ###12s block for MOTOR task
    print("%d Data samples with %d trials and %d features for each" % (sub_num, Trial_Num, Region_Num))

    fmri_data_matrix = []
    label_data_matrix = []
    for subi in range(Subject_Num):
        label_trial_data = np.array(subjects_trial_label_matrix[subi])
        condition_mask = pd.Series(label_trial_data).isin(target_name)
        ##condition_mask = pd.Series(label_trial_data).str.split('_',expand = True)[0].isin(target_name)
        fmri_data_matrix.append(subjects_tc_matrix[subi][condition_mask,:])
        label_data_matrix.append(label_trial_data[condition_mask])
    fmri_data_matrix = np.array(fmri_data_matrix).astype('float32',casting='same_kind')
    label_data_matrix = np.array(label_data_matrix)

    ##cut the trials into blocks
    chunks = int(np.floor(label_data_matrix.shape[-1] / block_dura))
    fmri_data_block = np.array(np.array_split(fmri_data_matrix, chunks, axis=1)).mean(axis=2).astype('float32',casting='same_kind')
    label_data_block = np.array(np.array_split(label_data_matrix, chunks, axis=1))[:, :, 0]

    ###reshape data to fit the model
    X_data = np.vstack(fmri_data_block[:, :sub_num, :]).astype('float32', casting='same_kind')
    Y_data = label_data_block[:, :sub_num].ravel()
    ###Y_data = np.vstack(np.repeat(np.expand_dims(label_data_block[:,:100],axis=2),Region_Num,axis=2))[:,0]
    print(X_data.shape, Y_data.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=10)
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype('float32', casting='same_kind')
    X_test = scaler.transform(X_test).astype('float32', casting='same_kind')
    n_classes = len(np.unique(label_data_block))
    le = preprocessing.LabelEncoder()
    le.fit(np.unique(Y_data))
    Y_train_int = le.transform(Y_train)
    Y_test_int = le.transform(Y_test)

    ########training a svm classifier
    clf2 = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo', random_state=10)
    scores = cross_val_score(clf2, X_train, Y_train_int, cv=5, scoring='accuracy')
    print("cross-validation scores:", scores)

    clf = svm.SVC(kernel='linear', decision_function_shape='ovo', random_state=10)
    clf.fit(X_train, Y_train_int)
    acc = metrics.accuracy_score(clf.predict(X_test), Y_test_int)

    clf.fit(X_train, Y_train_int)
    f1score = metrics.f1_score(clf.predict(X_test), Y_test_int, average='macro')
    print('Accuarcy on test data: %4f and f1-score %4f' % (acc, f1score))
    y_pred = clf.predict(X_test)
    print(metrics.classification_report(Y_test_int, y_pred, target_names=target_name))
    print('Confusion Matrix:')
    print(metrics.confusion_matrix(Y_test_int, y_pred, labels=range(n_classes)))

    if Flag_PCA:
        ##using pca for dimension reduction
        pca = PCA(n_components=my_comp, svd_solver='randomized', whiten=True)
        X_train_pca = pca.fit_transform(X_train)
        # clf = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo', random_state=10)
        scores_pca = cross_val_score(clf, X_train_pca, Y_train_int, cv=my_cv_fold, scoring='accuracy')
        print('SVM Scoring after PCA decomposition: ')
        print(scores_pca)

        ##using fastica for dimension reduction
        ica = FastICA(n_components=my_comp, whiten=True)
        X_train_ica = ica.fit_transform(X_train)
        # clf = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo', random_state=10)
        scores_ica = cross_val_score(clf, X_train_ica, Y_train_int, cv=my_cv_fold, scoring='accuracy')
        print('SVM Scoring after ICA decomposition: ')
        print(scores_ica)

        ##using kernelPCA for dimension reduction
        kpca = KernelPCA(n_components=my_comp, kernel='rbf')
        X_train_kpca = kpca.fit_transform(X_train)
        # clf = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo', random_state=10)
        scores_kpca = cross_val_score(clf, X_train_kpca, Y_train_int, cv=my_cv_fold, scoring='accuracy')
        print('SVM Scoring after DL decomposition: ')
        print(scores_kpca)

    return scores



def select_targetlabels_norest(fmri_data, label_data, target_labels=None):
    ###extract norest labels for classification
    if not target_labels:
        target_labels = list(np.unique(label_data))
        #target_labels.remove('rest')
        try:
            target_labels.remove('rest')
            print(target_labels)
        except:
            print('no label named rest')
            return fmri_data, label_data, target_labels

    try:
        condition_mask = pd.Series(label_data).isin(target_labels)
        #condition_mask = pd.Series(label_data).str.split('_', expand=True)[0].isin(target_labels)
    except:
        fmri_data = fmri_data.reshape(-1, fmri_data.shape[-1])
        label_data = label_data.flatten()
        condition_mask = pd.Series(label_data).isin(target_labels)
        #condition_mask = pd.Series(label_data).str.split('_', expand=True)[0].isin(target_labels)

    print(sum(condition_mask.astype(int))," types of task to be decoded")
    fmri_data_norest = np.array(fmri_data[condition_mask, :])
    label_data_norest = np.array(label_data[condition_mask])
    #print(fmri_data_norest.shape, label_data_norest.shape)

    return fmri_data_norest, label_data_norest

def subject_cross_validation_split(tc_matrix, label_matrix, sub_num=None, n_folds=10, testsize=0.2, valsize=0.2, randomseed=1234):

    Subject_Num, Trial_Num,Region_Num = np.array(tc_matrix).shape
    rs = np.random.RandomState(randomseed)
    if not sub_num or sub_num>Subject_Num:
        sub_num = Subject_Num

    train_sid_tmp, test_sid = train_test_split(range(sub_num), test_size=testsize, random_state=rs, shuffle=True)
    fmri_data_train = np.array(np.vstack([tc_matrix[i] for i in train_sid_tmp]))
    fmri_data_test = np.array(np.vstack([tc_matrix[i] for i in test_sid]))

    label_data_train = np.array([label_matrix[i] for i in train_sid_tmp])
    label_data_test = np.array([label_matrix[i] for i in test_sid]).reshape(len(test_sid) * Trial_Num, )

    scaler = preprocessing.StandardScaler().fit(fmri_data_train)
    X_train = scaler.transform(fmri_data_train).reshape(len(train_sid_tmp), Trial_Num, Region_Num)
    X_test = scaler.transform(fmri_data_test)
    Y_train = label_data_train.reshape((len(train_sid_tmp),Trial_Num))
    Y_test = label_data_test
    target_labels = list(np.unique(label_matrix))
    try:
        target_labels.remove('rest')
    except:
        print('no label named rest')

    X_test_norest, Y_test_norest = select_targetlabels_norest(X_test, Y_test, target_labels)
    nb_class = len(target_labels)

    from sklearn.model_selection import ShuffleSplit
    valsplit = ShuffleSplit(n_splits=n_folds, test_size=valsize, random_state=rs)
    X_train_scaled = []
    X_val_scaled = []
    Y_train_scaled = []
    Y_val_scaled = []
    for train_sid, val_sid in valsplit.split(train_sid_tmp):

        ##preprocess features and labels
        X = np.array(np.vstack([X_train[i] for i in train_sid]))
        Y = np.array(np.vstack([Y_train[i] for i in train_sid]))
        X_train_norest, Y_train_norest = select_targetlabels_norest(X, Y, target_labels)
        X_train_scaled.append(X_train_norest)
        Y_train_scaled.append(Y_train_norest)

        X = np.array(np.vstack([X_train[i] for i in val_sid]))
        Y = np.array(np.vstack([Y_train[i] for i in val_sid]))
        X_val_norest, Y_val_norest = select_targetlabels_norest(X, Y, target_labels)
        X_val_scaled.append(X_val_norest)
        Y_val_scaled.append(Y_val_norest)

        '''
        X_train_scaled.append(np.array(np.vstack([X_train[i] for i in train_sid])))
        X_val_scaled.append(np.array(np.vstack([X_train[i] for i in val_sid])))
        Y_train_scaled.append(np.ravel(np.array(np.vstack([Y_train[i] for i in train_sid]))))
        Y_val_scaled.append(np.ravel(np.array(np.vstack([Y_train[i] for i in val_sid]))))
        
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
    return X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled, X_test_norest, Y_test_norest


def my_svc_simple_subject_validation(subjects_tc_matrix,subjects_trial_label_matrix,sub_num=None,my_cv_fold=10,my_comp=20,my_testsize=0.2,my_valsize=0.2):
    ###using svm classification with subject-specific split of train, val and test

    Subject_Num, Trial_Num, Region_Num = np.array(subjects_tc_matrix).shape
    if Trial_Num != np.array(subjects_trial_label_matrix).shape[1]:
        print('Miss-matching trial infos for event and fmri data')
    if Subject_Num != np.array(subjects_trial_label_matrix).shape[0]:
        print('Adjust subject numbers for event data')
        subjects_trial_label_matrix = np.array(subjects_trial_label_matrix[:Subject_Num])
    else:
        subjects_trial_label_matrix = np.array(subjects_trial_label_matrix)

    if not sub_num or sub_num>Subject_Num:
        sub_num = Subject_Num

    ###SVM classifier
    ##split data into train, val and test in subject-level
    X_train, Y_train, X_val, Y_val, X_test, Y_test = subject_cross_validation_split(subjects_tc_matrix, subjects_trial_label_matrix,sub_num=sub_num,
                                                                                    n_folds=my_cv_fold, testsize=my_testsize, valsize=my_valsize)

    ##build a simple classifier using SVM
    print('Training the model using multi-class svc')
    clf = svm.SVC(decision_function_shape='ovr')
    train_acc = []
    val_acc = []
    for x_train, y_train, x_val, y_val, cvi in zip(X_train, Y_train, X_val, Y_val, range(my_cv_fold)):
        print('cv-fold: %d ...' % cvi)
        start_time = time.clock()
        clf.fit(x_train, y_train)
        train_acc.append(metrics.accuracy_score(clf.predict(x_train), y_train))
        val_acc.append(metrics.accuracy_score(clf.predict(x_val), y_val))
        print('Time Usage of model training in seconds: {} '.format(time.clock() - start_time))
    test_acc = metrics.accuracy_score(clf.predict(X_test), Y_test)
    scores = [np.mean(train_acc),np.mean(val_acc),test_acc]
    print('SVM Scoring:')
    print("Accuracy of prediction in training:{},validation:{} and testing:{}"
          .format(scores[0], scores[1], scores[2]))


    ##using pca for dimension reduction
    print('\n Dimensionality reduction using PCA before performing multi-class svc ')
    X_data_scaled = np.vstack([X_train[0], X_val[0]])
    pca = PCA(n_components=my_comp, svd_solver='randomized', whiten=True)
    pca.fit(X_data_scaled)
    train_acc = []
    val_acc = []
    for x_train, y_train, x_val, y_val, cvi in zip(X_train, Y_train, X_val, Y_val, range(my_cv_fold)):
        print('cv-fold: %d ...' % cvi)
        start_time = time.clock()
        x_train_pca = pca.fit_transform(x_train)
        x_val_pca = pca.fit_transform(x_val)
        clf.fit(x_train_pca, y_train)
        train_acc.append(metrics.accuracy_score(clf.predict(x_train_pca), y_train))
        val_acc.append(metrics.accuracy_score(clf.predict(x_val_pca), y_val))
        print('Time Usage of model training in seconds: {} '.format(time.clock() - start_time))
    X_test_pca = pca.fit_transform(X_test)
    test_acc = metrics.accuracy_score(clf.predict(X_test_pca), Y_test)
    scores_pca = [np.mean(train_acc),np.mean(val_acc),test_acc]
    print('SVM Scoring after PCA decomposition: ')
    print("Accuracy of prediction in training:{},validation:{} and testing:{}"
          .format(scores_pca[0], scores_pca[1], scores_pca[2]))

    ##using fastica for dimension reduction
    print('\n Dimensionality reduction using FastICA before performing multi-class svc ')
    ica = FastICA(n_components=my_comp, whiten=True)
    ica.fit(X_data_scaled)
    train_acc = []
    val_acc = []
    for x_train, y_train, x_val, y_val, cvi in zip(X_train, Y_train, X_val, Y_val, range(my_cv_fold)):
        print('cv-fold: %d ...' % cvi)
        start_time = time.clock()
        x_train_ica = ica.fit_transform(x_train)
        x_val_ica = ica.fit_transform(x_val)
        clf.fit(x_train_ica, y_train)
        train_acc.append(metrics.accuracy_score(clf.predict(x_train_ica), y_train))
        val_acc.append(metrics.accuracy_score(clf.predict(x_val_ica), y_val))
        print('Time Usage of model training in seconds: {} '.format(time.clock() - start_time))
    X_test_ica = ica.fit_transform(X_test)
    test_acc = metrics.accuracy_score(clf.predict(X_test_ica), Y_test)
    scores_ica = [np.mean(train_acc), np.mean(val_acc), test_acc]
    print('SVM Scoring after ICA decomposition: ')
    print("Accuracy of prediction in training:{},validation:{} and testing:{}"
          .format(scores_ica[0], scores_ica[1], scores_ica[2]))

    ##using kernelPCA for dimension reduction
    print('\n Dimensionality reduction using KernelPCA before performing multi-class svc ')
    kpca = KernelPCA(n_components=my_comp, kernel='rbf')
    kpca.fit(X_data_scaled)
    train_acc = []
    val_acc = []
    for x_train, y_train, x_val, y_val, cvi in zip(X_train, Y_train, X_val, Y_val, range(my_cv_fold)):
        print('cv-fold: %d ...' % cvi)
        start_time = time.clock()
        x_train_kpca = kpca.fit_transform(x_train)
        x_val_kpca = kpca.fit_transform(x_val)
        clf.fit(x_train_kpca, y_train)
        train_acc.append(metrics.accuracy_score(clf.predict(x_train_kpca), y_train))
        val_acc.append(metrics.accuracy_score(clf.predict(x_val_kpca), y_val))
        print('Time Usage of model training in seconds: {} '.format(time.clock() - start_time))
    X_test_kpca = kpca.fit_transform(X_test)
    test_acc = metrics.accuracy_score(clf.predict(X_test_kpca), Y_test)
    scores_kpca = [np.mean(train_acc), np.mean(val_acc), test_acc]
    print('SVM Scoring after DL decomposition: ')
    print("Accuracy of prediction in training:{},validation:{} and testing:{}"
          .format(scores_kpca[0], scores_kpca[1], scores_kpca[2]))

    return scores, scores_pca, scores_ica, scores_kpca


def subject_cross_validation_split_trials(tc_matrix, label_matrix,target_name, sub_num=None, block_dura=18, n_folds=10, testsize=0.2, valsize=0.1,randomseed=1234):
    ##randomseed=1234;testsize = 0.2;n_folds=10;valsize=0.1

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
    # print(fmri_data_block.shape,label_data_block.shape)

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

    print('Samples of Subjects for training: %d and testing %d and validating %d with %d classes' % (
    len(train_sid), len(test_sid), len(val_sid), nb_class))
    return X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled, X_test, Y_test


##########################################
def my_svc_simple_subject_validation_new(subjects_tc_matrix_new, subjects_trial_label_matrix_new, target_name, sub_num=None,
                                         block_dura=18, my_cv_fold=10,my_testsize=0.2, my_valsize=0.1):
    ##test for svm classification
    # my_cv_fold=10;my_testsize = 0.2;my_valsize=0.1;sub_num=100

    Subject_Num, Trial_Num, Region_Num = np.array(subjects_tc_matrix_new).shape
    if Trial_Num != np.array(subjects_trial_label_matrix_new).shape[1]:
        print('Miss-matching trial infos for event and fmri data')
    if Subject_Num != np.array(subjects_trial_label_matrix_new).shape[0]:
        print('Adjust subject numbers for event data')
        subjects_trial_label_matrix = np.array(subjects_trial_label_matrix_new[:Subject_Num])
    else:
        subjects_trial_label_matrix = np.array(subjects_trial_label_matrix_new)

    if not sub_num or sub_num>Subject_Num:
        sub_num = Subject_Num
    if not block_dura:
        block_dura = 18 ###12s block for MOTOR task

    ###SVM classifier
    ##split data into train, val and test in subject-level
    X_train, Y_train, X_val, Y_val, X_test, Y_test = \
        subject_cross_validation_split_trials(subjects_tc_matrix_new, subjects_trial_label_matrix, target_name,sub_num=sub_num,
                                              block_dura=block_dura,n_folds = my_cv_fold, testsize = my_testsize, valsize = my_valsize)

    ##build a simple classifier using SVM
    print('Training the model using multi-class svc')
    clf = svm.SVC(kernel='linear', decision_function_shape='ovo', random_state=10)
    train_acc = []
    val_acc = []
    for x_train, y_train, x_val, y_val, cvi in zip(X_train, Y_train, X_val, Y_val, range(my_cv_fold)):
        print('cv-fold: %d ...' % cvi)
        start_time = time.clock()
        clf.fit(x_train, y_train)
        train_acc.append(metrics.accuracy_score(clf.predict(x_train), y_train))
        val_acc.append(metrics.accuracy_score(clf.predict(x_val), y_val))
        # print('Time Usage of model training in seconds: {0:.2f} '.format(time.clock() - start_time))
    ##print(train_acc,val_acc)


    X_train_all = np.array(np.vstack((X_train[0], X_val[0])))
    Y_train_all = np.array(np.concatenate((Y_train[0], Y_val[0]), axis=0))
    print('sample size for training and testing: ', X_train_all.shape, Y_train_all.shape)
    le = preprocessing.LabelEncoder()
    le.fit(np.unique(Y_train_all))
    Y_train_int = le.transform(Y_train_all)
    Y_test_int = le.transform(Y_test)

    clf = svm.SVC(kernel='linear', decision_function_shape='ovo', random_state=10)
    clf.fit(X_train_all, Y_train_int)
    y_pred = clf.predict(X_test)
    print(metrics.classification_report(Y_test_int, y_pred, target_names=np.unique(Y_test)))
    print('Confusion Matrix:')
    print(metrics.confusion_matrix(Y_test_int, y_pred, labels=range(len(np.unique(Y_test)))))

    test_acc = metrics.accuracy_score(y_pred, Y_test_int)
    scores = np.array([np.mean(train_acc), np.mean(val_acc), test_acc]).astype('float32', casting='same_kind')
    print('SVM Scoring:')
    print("Accuracy of prediction in training:{},validation:{} and testing:{}"
          .format(scores[0], scores[1], scores[2]))

    return scores


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


def build_fc_nn_simple(subjects_tc_matrix,subjects_trial_label_matrix,target_name,layers=3,hidden_size=256,dropout=0.25,
                       batch_size=128,nepochs=100,testsize=0.1,valsize=0.1):
    ###classification using fully-connected neural networks
    Subject_Num, Trial_Num, Region_Num = np.array(subjects_tc_matrix).shape
    if Trial_Num != np.array(subjects_trial_label_matrix).shape[1]:
        print('Miss-matching trial infos for event and fmri data')
    if Subject_Num != np.array(subjects_trial_label_matrix).shape[0]:
        print('Adjust subject numbers for event data')
        subjects_trial_label_matrix = np.array(subjects_trial_label_matrix[:Subject_Num])
    else:
        subjects_trial_label_matrix = np.array(subjects_trial_label_matrix)

    ##feature matrix
    fmri_data = np.vstack(subjects_tc_matrix)
    label_data = subjects_trial_label_matrix.reshape(Subject_Num * Trial_Num, )
    #target_name = np.unique(list(task_contrasts.values()))  ##whether to exclude 'rest'
    condition_mask = pd.Series(label_data).isin(target_name)
    #condition_mask = pd.Series(label_data).str.split('_', expand=True)[0].isin(target_name)
    print(sum(condition_mask.astype(int))," types of task to be decoded")

    X_data = fmri_data[condition_mask,]
    Y_data = label_data[condition_mask]

    X_data_scaled = preprocessing.scale(X_data)  # with zero mean and unit variance.
    le = preprocessing.LabelEncoder()
    le.fit(target_name)
    Y_data_int = le.transform(Y_data)
    Y_data_label = np_utils.to_categorical(Y_data_int)

    print("%d Data samples with %d features and output in %d classes" %
          (X_data_scaled.shape[0],X_data_scaled.shape[1], Y_data_label.shape[1]))
    '''
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
    '''
    fc_model = model.build_fc_nn_model(Region_Num,len(target_name),layers=layers,hidden_size=hidden_size,dropout=dropout)
    X_train, X_test, Y_train, Y_test = train_test_split(X_data_scaled, Y_data_label, test_size=testsize, random_state=10)
    print('Samples for training: %d and testing %d with %d features' % (X_train.shape[0], X_test.shape[0], X_train.shape[1]))

    ####running the model
    start_time = time.clock()
    model_history = fc_model.fit(X_train, Y_train, batch_size=batch_size, epochs=nepochs, validation_split=valsize)
    ##plot_history(model_history)
    print('Time Usage in seconds: {}'.format(time.clock() - start_time))
    train_loss = model_history.history['loss'][-1]
    train_acc = model_history.history['acc'][-1]
    val_loss = model_history.history['val_loss'][-1]
    val_acc = model_history.history['val_acc'][-1]

    test_loss, test_mse, test_acc = fc_model.evaluate(X_test, Y_test)
    print('Testing sets: Loss: %4f and Accuracy: %4f' % (test_loss, test_acc))
    return test_acc, val_acc, train_acc


def build_fc_nn_subject_validation(subjects_tc_matrix,subjects_trial_label_matrix,target_name,flag_trialavg=1,block_dura=18,layers=3,hidden_size=256, dropout=0.25,learning_rate=0.001, batch_size=128,nepochs=20,my_cv_fold=10,testsize=0.2,valsize=0.2):
    ###classification using fully-connected neural networks with subject-specific split of train, val and test

    Subject_Num, Trial_Num, Region_Num = np.array(subjects_tc_matrix).shape
    if Trial_Num != np.array(subjects_trial_label_matrix).shape[1]:
        print('Miss-matching trial infos for event and fmri data')
    if Subject_Num != np.array(subjects_trial_label_matrix).shape[0]:
        print('Adjust subject numbers for event data')
        subjects_trial_label_matrix = np.array(subjects_trial_label_matrix[:Subject_Num])
    else:
        subjects_trial_label_matrix = np.array(subjects_trial_label_matrix)
   
    if not block_dura:
        block_dura = 18 ###12s block for MOTOR task
    if not learning_rate:
        learning_rate = 0.001 ###12s block for MOTOR task


    ######fully-connected neural networks
    fc_model = model.build_fc_nn_model(Region_Num, len(target_name), layers=layers, hidden_size=hidden_size,dropout=dropout)
    adam = Adam(lr=learning_rate)
    fc_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    if not flag_trialavg:
        ##split data into train, val and test in subject-level
        X_train, Y_train, X_val, Y_val, X_test, Y_test = subject_cross_validation_split(subjects_tc_matrix, subjects_trial_label_matrix, n_folds=my_cv_fold, testsize=testsize, valsize=valsize)
        print('Samples for training: %d and testing %d with %d features' % (X_train[0].shape[0], X_test.shape[0], X_train[0].shape[1]))
    else:
        X_train, Y_train, X_val, Y_val, X_test, Y_test = \
        subject_cross_validation_split_trials(subjects_tc_matrix, subjects_trial_label_matrix, target_name,block_dura=block_dura,
                                              n_folds = my_cv_fold, testsize = testsize, valsize = valsize)
    le = preprocessing.LabelEncoder()
    le.fit(target_name)
    nb_class = len(target_name)
    Y_test_label = np_utils.to_categorical(le.transform(Y_test),num_classes=nb_class)
    
    ####running the model
    start_time = time.clock()
    ##build a simple classifier using SVM
    print('Training the model using cross-validation')
    clf = svm.SVC(kernel='linear', decision_function_shape='ovo', random_state=10)
    train_acc_cv = []
    val_acc_cv = []
    for x_train, y_train, x_val, y_val, cvi in zip(X_train, Y_train, X_val, Y_val, range(my_cv_fold)):
        print('cv-fold: %d ...' % cvi)
        start_time = time.clock()
        y_train_label = np_utils.to_categorical(le.transform(y_train),num_classes=nb_class)
        y_val_label = np_utils.to_categorical(le.transform(y_val),num_classes=nb_class)
        model_history = fc_model.fit(x_train, y_train_label, batch_size=batch_size, epochs=nepochs, validation_data=(x_val,y_val_label))
        ##plot_history(model_history)
        print('Time Usage in seconds: {}'.format(time.clock() - start_time))
        train_loss = model_history.history['loss'][-1]
        train_acc = model_history.history['acc'][-1]
        val_loss = model_history.history['val_loss'][-1]
        val_acc = model_history.history['val_acc'][-1]
        train_acc_cv.append(train_acc)
        val_acc_cv.append(val_acc)
        # print('Time Usage of model training in seconds: {0:.2f} '.format(time.clock() - start_time))
    print(train_acc_cv,val_acc_cv)
    
   
    X_train_all = np.array(np.vstack((X_train[0], X_val[0])))
    Y_train_all = np.array(np.concatenate((Y_train[0], Y_val[0]), axis=0))
    Y_train_label = np_utils.to_categorical(le.transform(Y_train_all),num_classes=nb_class)
    Y_test_int = le.transform(Y_test)
    print('sample size for training and testing: ', X_train_all.shape, Y_train_all.shape)

    model_history = fc_model.fit(X_train_all, Y_train_label, batch_size=batch_size, epochs=100,validation_split=valsize)
    y_pred = [np.argmax(y, axis=None, out=None) for y in fc_model.predict(X_test)]
    print(np.array(y_pred).shape,Y_test_int.shape)
    #test_loss, test_mse, test_acc = fc_model.evaluate(X_test, Y_test_int)
    #print('Testing sets: Loss: %4f and Accuracy: %4f' % (test_loss, test_acc))
    print(metrics.classification_report(Y_test_int, y_pred, target_names=np.unique(Y_test)))
    print('Confusion Matrix:')
    print(metrics.confusion_matrix(Y_test_int, y_pred, labels=range(len(np.unique(Y_test)))))

    test_acc = metrics.accuracy_score(Y_test_int, y_pred)
    scores = np.array([np.mean(train_acc_cv), np.mean(val_acc_cv), test_acc]).astype('float32', casting='same_kind')
    print('Fully-connected neural network Scoring:')
    print("Accuracy of prediction in training:{},validation:{} and testing:{}"
          .format(scores[0], scores[1], scores[2]))

    return test_acc, val_acc, train_acc

class hcp_2dcnn_fmri(object):
    """define graph convolution parameters including Laplacian, filters and coarsening
        """

    def __init__(self, config):
        self.config = config
        self.modality = config.modality
        self.task_contrasts = config.task_contrasts
        self.EVS_files = config.EVS_files  ##using the same ev-design files but loading different fmri data
        self.nlabels = len(np.unique(list(config.task_contrasts.values())))

    def load_fmri_data(self, label_matrix, fmri_files=None, confound_files=None):
        pathfmri = self.config.pathfmri
        startsub = self.config.startsub
        endsub = self.config.endsub
        modality = self.modality

        if not fmri_files:
            fmri_files = self.config.fmri_files
        if not confound_files:
            confound_files = self.config.confound_files

        ###collect fmri nifti files from fmri folders
        print('Collecting fmri files from subject {} to {} from folder:{}'.format(startsub, endsub, pathfmri))

        Subject_Num = len(fmri_files)
        fmri_files_Img = []
        for subj in range(Subject_Num):
            pathsub = Path(os.path.dirname(fmri_files[subj]))
            ###loading the nifti files from the same set of subjects
            if not os.path.isfile(str(sorted(pathsub.glob("_".join(["tfMRI", modality]) + "*.nii.gz"))[0])):
                print("Warnings: NIFTI file not exist for subject: %s" % pathsub)
                break
            fmri_files_Img.append(str(sorted(pathsub.glob("_".join(["tfMRI", modality]) + "*.nii.gz"))[0]))

        print('%d subjects included in the dataset' % len(fmri_files_Img))
        if len(fmri_files_Img) != len(fmri_files):
            print('Mismatching number of fmri data files! Please check the data in %s/  !' % str(pathfmri))
        #print(fmri_files_Img)


        self.fmri_files_Img = fmri_files_Img
        self.confound_files = confound_files
        self.Subject_Num = len(fmri_files_Img)
        tc_matrix = nib.load(fmri_files_Img[0])
        self.Trial_Num = tc_matrix.shape[-1]   ##all fmri data from different subjects have the same length in time
        print("Data samples including %d subjects with %d trials " % (self.Subject_Num, self.Trial_Num))

        ####get all 2d fmri data
        try:
            print(label_matrix.loc[0, :])
        except:
            label_matrix = pd.DataFrame(data=np.array(label_matrix), columns=['trial' + str(i + 1) for i in range(self.Trial_Num)])

        fmri_data_cnn = []
        label_data_cnn = []
        for subj in np.arange(self.Subject_Num):
            trial_mask = pd.Series(label_matrix.loc[subj, :]).isin(np.unique(list(self.task_contrasts.values())))  ##['hand', 'foot','tongue']

            fmri_img_trial = image.index_img(fmri_files_Img[subj], np.where(trial_mask)[0])
            ###use each slice along z-axis as one sample
            fmri_data_trial = fmri_img_trial.get_data()
            label_data_trial = np.array(label_matrix.loc[subj, trial_mask])

            fmri_data_cnn.append(fmri_data_trial)
            label_data_cnn.append(label_data_trial)

        return fmri_data_cnn, label_data_cnn


    def subject_cross_validation_split(self, fmri_data_cnn, label_data_cnn, target_labels=None, n_folds=20, testsize=0.2, valsize=0.2, randomseed=1234):

        from sklearn.model_selection import train_test_split
        from sklearn import preprocessing
        if not target_labels:
            target_labels = list(np.unique(list(self.task_contrasts.values())))
        try:
            target_labels.remove('rest')
            print(target_labels)
        except:
            print('no label named rest')
        nb_class = len(target_labels)

        img_rows, img_cols, img_deps = fmri_data_cnn[0].shape[:-1]
        Subject_Num, Trial_Num = np.array(label_data_cnn).shape

        rs = np.random.RandomState(randomseed)
        train_sid_tmp, test_sid = train_test_split(range(Subject_Num), test_size=testsize, random_state=rs,shuffle=True)
        fmri_data_cnn_test = np.array(fmri_data_cnn[test_sid])
        fmri_data_cnn_test = np.vstack(np.transpose(fmri_data_cnn_test.reshape((len(test_sid),img_rows, img_cols, np.prod(fmri_data_cnn[0].shape[2:]))), (0, 3, 1, 2)))
        label_data_cnn_test = np.vstack(np.repeat(label_data_cnn[test_sid], img_deps, axis=0)).flatten()

        fmri_data_train = np.array(fmri_data_cnn[train_sid_tmp])
        fmri_data_train = np.vstack(np.transpose(fmri_data_train.reshape((len(train_sid_tmp), img_rows, img_cols, np.prod(fmri_data_cnn[0].shape[2:]))),(0, 3, 1, 2)))
        label_data_train = np.vstack(np.repeat(label_data_cnn[train_sid_tmp], img_deps, axis=1)).flatten()

        ##preprocess input features and output labels
        scaler = preprocessing.StandardScaler().fit(fmri_data_train)
        X_train = np.expand_dims(scaler.transform(fmri_data_train), 3)  #
        X_test = np.expand_dims(scaler.transform(fmri_data_cnn_test), 3)
        le = preprocessing.LabelEncoder()
        le.fit(target_labels)
        Y_train = np_utils.to_categorical(le.transform(label_data_train))
        Y_test = np_utils.to_categorical(le.transform(label_data_cnn_test))


        from sklearn.model_selection import ShuffleSplit
        valsplit = ShuffleSplit(n_splits=n_folds, test_size=valsize, random_state=rs)
        X_train_scaled = []
        X_val_scaled = []
        Y_train_scaled = []
        Y_val_scaled = []
        for train_sid, val_sid in valsplit.split(train_sid_tmp):
            ##preprocess features and labels
            X_train_scaled.append(np.array(np.vstack([X_train[i] for i in train_sid])))
            Y_train_scaled.append(np.array(np.vstack([Y_train[i] for i in train_sid])))

            X_val_scaled.append(np.array(np.vstack([X_train[i] for i in val_sid])))
            Y_val_scaled.append(np.array(np.vstack([Y_train[i] for i in val_sid])))

        print('Samples for training: %d and testing %d and validating %d with %d classes' % (len(train_sid), len(test_sid), len(val_sid), nb_class))
        return X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled, X_test, Y_test


    def build_2dcnn_subject_validation(self,subjects_trial_label_matrix,filters=32, convsize=3,poolsize=2,conv_layers=2, hidden_size=256,
                                       batch_size=128,nepochs=100, my_cv_fold=10,testsize=0.2,valsize=0.2):

        ##pre-settings of the analysis
        target_name = np.unique(list(self.task_contrasts.values()))
        nb_class = len(target_name)
        print('Processing with the task_modality: %s with %d subtypes' % (self.modality, len(target_name)))
        print(target_name)

        ##collecting the fmri files
        fmri_data_cnn, label_data_cnn = self.load_fmri_data(subjects_trial_label_matrix)
        print('including %d fmri files and %d confounds files \n\n' % (len(self.fmri_files_Img), len(self.confound_files)))

        print('Preparing fmri data before model training  ...')
        X_train, Y_train, X_val, Y_val, X_test, Y_test = subject_cross_validation_split(fmri_data_cnn, label_data_cnn, n_folds=my_cv_fold, testsize=testsize, valsize=valsize)
        print(np.array(X_train).shape, np.array(X_val).shape, np.array(X_test).shape, np.array(Y_train).shape)
        print(np.unique(Y_test))
        print('Samples for training: %d and testing %d and validating %d with %d classes' % (len(X_train), len(X_test), len(X_val), nb_class))

        ####model training and testing
        print('Training a 2D-CNN model for classification ...')
        import keras.backend as K
        import model as my_model

        img_rows, img_cols, img_deps = fmri_data_cnn[0].shape[:-1]
        img_shape = []
        if K.image_data_format() == 'channels_first':
            img_shape = (1, img_rows, img_cols)
        elif K.image_data_format() == 'channels_last':
            img_shape = (img_rows, img_cols, 1)

        model_test = my_model.build_cnn_model(img_shape, nb_class, filters=filters, convsize=convsize, poolsize=poolsize, hidden_size=hidden_size, conv_layers=conv_layers)

        ##initalization
        train_acc = []
        train_loss = []
        test_acc = []
        test_loss = []
        val_acc = []
        val_loss = []
        ###subject-specific cross-validation
        for x_train, y_train, x_val, y_val, tcount in zip(X_train, Y_train, X_val, Y_val, range(len(X_train))):
            print('\nFold #%d: training on %d samples with %d features, validating on %d samples and testing on %d samples' %
                  (tcount + 1, x_train.shape[0], x_train.shape[1], x_val.shape[0], X_test.shape[0]))

            model_test_history = model_test.fit(x_train, y_train, batch_size=batch_size, epochs=nepochs,validation_data=(x_val, y_val))
            # plot_history(model_test_history)
            train_acc.append(model_test_history.history['acc'])
            train_loss.append(model_test_history.history['loss'])
            val_acc.append(model_test_history.history['val_acc'])
            val_loss.append(model_test_history.history['val_loss'])

            loss, acc = model_test.evaluate(X_test, Y_test, verbose=1)
            test_acc.append(acc)
            test_loss.append(loss)

        print(train_acc, test_acc, val_acc)

        ###save the trained model
        # serialize model to JSON
        print("Saved 2d-cnn model to disk")
        pathout = self.config.pathout
        modality = self.modality
        model_filename = pathout + modality + "_model_test_2dcnn.h5"
        model_test_json = model_test.to_json()
        with open(model_filename[:-2]+"json", "w") as json_file:
            json_file.write(model_test_json)
        # serialize weights to HDF5
        model_test.save_weights(model_filename)

        return train_acc, test_acc, val_acc



class hcp_gcnn_fmri(object):
    """define graph convolution parameters including Laplacian, filters and coarsening
        """

    def __init__(self, config):
        #self.config = config
        self.modality = config.modality
        self.task_contrasts = config.task_contrasts
        self.Trial_dura = config.Trial_dura
        self.Subject_Num = config.Subject_Num
        #self.gcnn_flag = config.FLAGS
        self.parcel_map = config.mmp_atlas
        self.adj_mat_type = config.gcnn_adj_mat_type
        self.adj_mat_file = config.gcnn_adj_mat_dict[config.gcnn_adj_mat_type]
        self.coarsening_levels = config.gcnn_coarsening_levels
        self.nlabels = len(np.unique(list(config.task_contrasts.values())))


    def gccn_model_common_param(self, training_samples, block_dura=1, layers=3,hidden_size=256,pool_size=4,batch_size=128,nepochs=100):
        ##first load adjacent matrix to build the initial graph
        try:
            #import cnn_graph
            from cnn_graph.lib import models, graph, coarsening, utils
        except ImportError:
            print('Could not find the package of graph-cnn ...')
            print('Please check the location where cnn_graph is !\n')

        Subject_Num = self.Subject_Num
        modality = self.modality
        C = self.nlabels + 1

        ##model params
        gcnn_common = {}
        gcnn_common['dir_name'] = modality + '/' + 'win' + str(block_dura) + '/'
        gcnn_common['num_epochs'] = nepochs
        gcnn_common['batch_size'] = batch_size
        gcnn_common['decay_steps'] = training_samples / gcnn_common['batch_size'] ##refine this according to samples
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
        gcnn_common['F'] = [32*math.pow(2, li) for li in range(layers)]   #[32, 64, 128]  # Number of graph convolutional filters.
        gcnn_common['K'] = [25 for li in range(layers)]    #[25, 25, 25]  # Polynomial orders.
        gcnn_common['p'] = [pool_size for li in range(layers)]   #[4, 4, 4]  # Pooling sizes.
        gcnn_common['M'] = [hidden_size, C]  # Output dimensionality of fully connected layers.

        self.gcnn_common = gcnn_common
        self.params = {}
        return self.gcnn_common

    def build_fc_cnn(self, Laplacian_list=None):
        from cnn_graph.lib import models
        #self.gccn_model_common_param()
        common = self.gcnn_common

        ##model1: no convolution
        name = 'softmax'
        params = common.copy()
        params['dir_name'] += name
        params['F'] = []
        params['K'] = []
        params['p'] = []
        print(params)
        self.params[name] = params

        if not Laplacian_list:
            print('Laplacian matrix for multi-scale graphs are requried!')
        model = models.cgcnn(Laplacian_list, **params)
        print('Using no convolutional layers and directly on classification\n')
        return model, name

    def build_fourier_graph_cnn(self, Laplacian_list=None):
        from cnn_graph.lib import models
        #self.gccn_model_common_param()
        common = self.gcnn_common

        if not Laplacian_list:
            print('Laplacian matrix for multi-scale graphs are requried!')
        else:
            print('Laplacian matrix for multi-scale graphs:')
            print([Laplacian_list[li].shape for li in range(len(Laplacian_list))])

        ##model#1: two convolutional layers with fourier transform as filters
        name = 'fgconv_fgconv_fc_softmax'  # 'Non-Param'
        params = common.copy()
        params['dir_name'] += name
        params['filter'] = 'fourier'
        params['K'] = np.zeros(len(common['p']), dtype=int)
        for pi, li in zip(common['p'], range(len(common['p']))):
            if pi == 2:
                params['K'][li] = Laplacian_list[li].shape[0]
            if pi == 4:
                params['K'][li] = Laplacian_list[li * 2].shape[0]

        print(params)
        self.params[name] = params

        model = models.cgcnn(Laplacian_list, **params)
        print('Building convolutional layers with fourier basis of Laplacian\n')
        print([Laplacian_list[li].shape for li in range(len(Laplacian_list))])
        return model, name


    def build_spline_graph_cnn(self, Laplacian_list=None):
        from cnn_graph.lib import models
        #self.gccn_model_common_param()
        common = self.gcnn_common

        if not Laplacian_list:
            print('Laplacian matrix for multi-scale graphs are requried!')
        else:
            print('Laplacian matrix for multi-scale graphs:')
            print([Laplacian_list[li].shape for li in range(len(Laplacian_list))])

        ##model#2: two convolutional layers with spline basis as filters
        name = 'sgconv_sgconv_fc_softmax'  # 'Non-Param'
        params = common.copy()
        params['dir_name'] += name
        params['filter'] = 'spline'
        print(params)
        self.params[name] = params

        model = models.cgcnn(Laplacian_list, **params)
        print('Building convolutional layers with spline basis\n')
        print([Laplacian_list[li].shape for li in range(len(Laplacian_list))])
        return model, name

    def build_chebyshev_graph_cnn(self, Laplacian_list=None):
        from cnn_graph.lib import models
        #self.gccn_model_common_param()
        common = self.gcnn_common
        common['learning_rate'] = 0.005  # 0.03 in the paper but sgconv_sgconv_fc_softmax has difficulty to converge
        common['decay_rate'] = 0.9  ##0.95

        if not Laplacian_list:
            print('Laplacian matrix for multi-scale graphs are requried!')
        else:
            print('Laplacian matrix for multi-scale graphs:')
            print([Laplacian_list[li].shape for li in range(len(Laplacian_list))])

        ##model#3: two convolutional layers with Chebyshev polynomial as filters
        name = 'cgconv_cgconv_fc_softmax'  # 'Non-Param'
        params = common.copy()
        params['dir_name'] += name
        params['filter'] = 'chebyshev5'
        print(params)
        self.params[name] = params

        model = models.cgcnn(Laplacian_list, **params)
        print('Building convolutional layers with Chebyshev polynomial\n')
        print([Laplacian_list[li].shape for li in range(len(Laplacian_list))])
        return model, name

    def build_graph_adj_mat(self, Nneighbours=8, noise_level=0):
        try:
            #import cnn_graph
            from cnn_graph.lib import models, graph, coarsening, utils
        except ImportError:
            print('Could not find the package of graph-cnn ...')
            print('Please check the location where cnn_graph is !\n')

        ##loading the first-level graph adjacent matrix based on surface neighbourhood
        adjacent_mat_file = self.adj_mat_file
        adjacent_mat_type = self.adj_mat_type
        if adjacent_mat_type.lower() == 'surface':
            print('Loading adjacent matrix based on counting connected vertices between parcels')

            adj_mat = nib.load(adjacent_mat_file).get_data()
            adj_mat = sparse.csr_matrix(adj_mat)
        elif adjacent_mat_type.lower() == 'sc':
            print('Calculate adjacent graph based on structural covaraince of corrThickness across subjects')
            conn_matrix = nib.load(adjacent_mat_file).get_data()
            atlas_roi = nib.load(self.parcel_map).get_data()
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

    def show_gcn_results(self, s, fontsize=None):

        if fontsize:
            import matplotlib.pyplot as plt
            plt.rc('pdf', fonttype=42)
            plt.rc('ps', fonttype=42)
            plt.rc('font', size=fontsize)  # controls default text sizes
            plt.rc('axes', titlesize=fontsize)  # fontsize of the axes title
            plt.rc('axes', labelsize=fontsize)  # fontsize of the x any y labels
            plt.rc('xtick', labelsize=fontsize)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=fontsize)  # fontsize of the tick labels
            plt.rc('legend', fontsize=fontsize)  # legend fontsize
            plt.rc('figure', titlesize=fontsize)  # size of the figure title
        print('  accuracy        F1             loss        time [ms]  name')
        print('test  train   test  train   test     train')
        for name in sorted(s.names):
            print('{:5.2f} {:5.2f}   {:5.2f} {:5.2f}   {:.2e} {:.2e}   {:3.0f}   {}'.format(
                s.test_accuracy[name], s.train_accuracy[name],
                s.test_f1[name], s.train_f1[name],
                s.test_loss[name], s.train_loss[name], s.fit_time[name] * 1000, name))
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


    def build_graph_cnn_subject_validation(self,subjects_tc_matrix,subjects_trial_label_matrix,target_name,block_dura=1,layers=3,hidden_size=256,pool_size=4,batch_size=128,nepochs=100,my_cv_fold=10,testsize=0.2,valsize=0.1):
        ###classification using graph convolution neural networks with subject-specific split of train, val and test
        Subject_Num, Trial_Num, Region_Num = np.array(subjects_tc_matrix).shape
        if Trial_Num != np.array(subjects_trial_label_matrix).shape[1]:
            print('Miss-matching trial infos for event and fmri data')
        if Subject_Num != np.array(subjects_trial_label_matrix).shape[0]:
            print('Adjust subject numbers for event data')
            subjects_trial_label_matrix = np.array(subjects_trial_label_matrix[:Subject_Num])
        else:
            subjects_trial_label_matrix = np.array(subjects_trial_label_matrix)

        ##split data into train, val and test in subject-level
        X_train, Y_train, X_val, Y_val, X_test, Y_test = \
            subject_cross_validation_split_trials(subjects_tc_matrix, subjects_trial_label_matrix, target_name,sub_num=Subject_Num,
                                                  block_dura=block_dura, n_folds=my_cv_fold, testsize=testsize,valsize=valsize)
        # X_train, Y_train, X_val, Y_val, X_test, Y_test = \
        #    subject_cross_validation_split_trials_new(subjects_tc_matrix, subjects_trial_label_matrix, target_name,sub_num=Subject_Num,
        #                                              block_dura=block_dura,n_folds = my_cv_fold, testsize = testsize, valsize = valsize)


        ###start of graph cnn codes
        ##first load adjacent matrix to build the initial graph
        try:
            #import cnn_graph
            from cnn_graph.lib import models, graph, coarsening, utils
        except ImportError:
            print('Could not find the package of graph-cnn ...')
            print('Please check the location where cnn_graph is !\n')

        ##loading the first-level graph adjacent matrix based on surface neighbourhood
        '''
        adjacent_mat_file = self.adj_mat  ##self.gcnn_flag.adj_mat
        adj_mat = nib.load(adjacent_mat_file).get_data()
        print(adj_mat.shape)
        adj_mat = sparse.csr_matrix(adj_mat)
        A = graph.replace_random_edges(adj_mat, 0.01)
        '''
        A = self.build_graph_adj_mat(Nneighbours=8, noise_level=0.001)

        ###build multi-level graph using coarsen (div by 2 at each level)
        coarsening_levels = self.coarsening_levels
        graphs, perm = coarsening.coarsen(A, levels=coarsening_levels, self_connections=False)
        L = [graph.laplacian(A, normalized=True) for A in graphs]
        #graph.plot_spectrum(L)

        ###pre-setting of common parameters
        self.nlabels = len(target_name)
        self.gccn_model_common_param(X_train[0].shape[0], layers=layers, hidden_size=hidden_size, pool_size=pool_size, batch_size=batch_size, nepochs=nepochs)
        common = self.gcnn_common

        model_perf = utils.model_perf()
        model1, gcnn_name1 = self.build_fc_cnn(Laplacian_list=L)
        model2, gcnn_name2 = self.build_fourier_graph_cnn(Laplacian_list=L)
        model3, gcnn_name3 = self.build_spline_graph_cnn(Laplacian_list=L)
        model4, gcnn_name4 = self.build_chebyshev_graph_cnn(Laplacian_list=L)
        gcnn_model_dicts = {#gcnn_name1: model1,
                            #gcnn_name2: model2,
                            gcnn_name3: model3,
                            gcnn_name4: model4}

        ##initalization
        train_acc = {};
        train_loss = {};
        test_acc = {};
        test_loss = {};
        val_acc = {};
        val_loss = {};
        accuracy = {};
        loss = {};
        t_step = {};
        for name in gcnn_model_dicts.keys():
            train_acc[name] = []
            train_loss[name] = []
            test_acc[name] = []
            test_loss[name] = []
            val_acc[name] = []
            val_loss[name] = []
            accuracy[name] = []
            loss[name] = []
            t_step[name] = []

        ###subject-specific cross-validation
        d = {k: v + 1 for v, k in enumerate(sorted(set(Y_test)))}
        test_labels = np.array([d[x] for x in Y_test])
        print(np.unique(Y_test))

        for x_train, y_train, x_val, y_val, tcount in zip(X_train, Y_train, X_val, Y_val, range(len(X_train))):
            train_data = coarsening.perm_data(x_train.reshape(-1, Region_Num), perm)
            train_labels = np.array([d[x] for x in y_train])
            val_data = coarsening.perm_data(x_val.reshape(-1, Region_Num), perm)
            val_labels = np.array([d[x] for x in y_val])
            test_data = coarsening.perm_data(X_test.reshape(-1, Region_Num), perm)
            print('\nFold #%d: training on %d samples with %d features, validating on %d samples and testing on %d samples' %
                (tcount + 1, train_data.shape[0], train_data.shape[1], val_data.shape[0], test_data.shape[0]))

            for name, model in gcnn_model_dicts.items():
                print('Training graph cnn using %s filters!' % name)

                ###training
                acc, los, tstep = model.fit(train_data, train_labels, val_data, val_labels)
                accuracy[name].append(acc)
                loss[name].append(los)
                t_step[name].append(t_step)

                ##evaluation
                params = self.params[name]  ###params should be included in the model already; need to fix this
                print(name,params)
                model_perf.test(model, name, params,
                                train_data, train_labels, val_data, val_labels, test_data, test_labels)
                train_acc[name].append(model_perf.train_accuracy[name])
                train_loss[name].append(model_perf.train_loss[name])
                test_acc[name].append(model_perf.test_accuracy[name])
                test_loss[name].append(model_perf.test_loss[name])
                val_acc[name].append(model_perf.fit_accuracies[name])
                val_loss[name].append(model_perf.fit_losses[name])

                print('Accuracy of training:{},testing:{}'.format(np.mean(train_acc[name]), np.mean(test_acc[name])))
                print('Accuracy of validation:', np.mean(np.max(val_acc[name], axis=1)))
                print('\n')
        #print(accuracy, loss)
        print(train_acc, test_acc, val_acc)


        '''
        ###training figures
        print(accuracy, loss)
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(np.mean(accuracy, axis=0), 'b.-')
        ax1.set_ylabel('validation accuracy', color='b')
        ax2 = ax1.twinx()
        ax2.plot(np.mean(loss, axis=0), 'g.-')
        ax2.set_ylabel('training loss', color='g')
        plt.show()
        '''
        ###summarize the results
        model_perf.show()

        return train_acc, test_acc, val_acc


    def subject_cross_validation_split_trials_new(self, tc_matrix, label_matrix, target_name, sub_num=None,block_dura=1,
                                                  n_folds=10, testsize=0.2, valsize=0.1, randomseed=1234):
        ##randomseed=1234;testsize = 0.2;n_folds=10;valsize=0.1
        from sklearn import preprocessing
        from sklearn.model_selection import cross_val_score, train_test_split, ShuffleSplit

        Subject_Num, Trial_Num, Region_Num = np.array(tc_matrix).shape
        rs = np.random.RandomState(randomseed)
        if not sub_num or sub_num > Subject_Num:
            sub_num = Subject_Num

        ##block_dura = self.block_dura   ##global setting for all methods
        modality = self.modality
        Trial_dura = self.Trial_dura
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
                fmri_data_block = np.array(np.vstack(np.array_split(fmri_data_block, chunks, axis=1)[:-1])).mean(axis=1).astype('float32',
                                                                                                                                casting='same_kind')
                label_data_trial_block = np.array(np.vstack(np.array_split(label_data_trial_block, chunks, axis=1)[:-1]))[:, 0]
            else:
                fmri_data_block = np.array(np.vstack(np.array_split(fmri_data_block, chunks, axis=1))).mean(axis=1).astype('float32',
                                                                                                                           casting='same_kind')
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
        if len(train_sid_tmp) < 2 or len(test_sid) < 2:
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


    def build_graph_cnn_subject_validation_new(self, subjects_tc_matrix, subjects_trial_label_matrix, target_name,block_dura=1,
                                               layers=3, pool_size=4, hidden_size=256, batch_size=128, nepochs=100,
                                               my_cv_fold=10, testsize=0.2, valsize=0.2):
        ###classification using graph convolution neural networks with subject-specific split of train, val and test
        Subject_Num = np.array(subjects_tc_matrix).shape[0]
        Trial_Num, Region_Num = np.array(subjects_tc_matrix[0]).shape
        if Trial_Num != np.array(subjects_trial_label_matrix).shape[1]:
            print('Miss-matching trial infos for event and fmri data')
        if Subject_Num != np.array(subjects_trial_label_matrix).shape[0]:
            print('Adjust subject numbers for event data')
            print('Need to run preclean_data before to ensure size matching between fmri and event data!')
            Subject_Num = min(np.array(subjects_tc_matrix).shape[0], np.array(subjects_trial_label_matrix).shape[0])
            subjects_trial_label_matrix = np.array(subjects_trial_label_matrix[:Subject_Num])
        else:
            subjects_trial_label_matrix = np.array(subjects_trial_label_matrix)

        ##split data into train, val and test in subject-level
        X_train, Y_train, X_val, Y_val, X_test, Y_test = \
            self.subject_cross_validation_split_trials_new(subjects_tc_matrix, subjects_trial_label_matrix, target_name,block_dura=block_dura,
                                                           n_folds=my_cv_fold, testsize=testsize, valsize=valsize)

        X_train_all = np.array(np.vstack((X_train[0], X_val[0])))
        Y_train_all = np.array(np.concatenate((Y_train[0], Y_val[0]), axis=0))
        print('sample size for training and testing: ', X_train_all.shape, Y_train_all.shape)

        ##################################################################
        ###prepare for gcn model
        ##first load adjacent matrix to build the initial graph
        try:
            #import cnn_graph
            from cnn_graph.lib import models, graph, coarsening, utils
        except ImportError:
            print('Could not find the package of graph-cnn ...')
            print('Please check the location where cnn_graph is !\n')

        ###pre-setting of common parameters
        modality = self.modality
        adj_mat_file = self.adj_mat_file
        adj_mat_type = self.adj_mat_type
        coarsening_levels = self.coarsening_levels
        gcnn_common = self.gccn_model_common_param(X_train[0].shape[0], block_dura=block_dura, layers=layers, pool_size=pool_size, hidden_size=hidden_size, batch_size=batch_size,nepochs=nepochs)

        A = self.build_graph_adj_mat(adj_mat_file, adj_mat_type)
        ###build multi-level graph using coarsen (div by 2 at each level)
        graphs, perm = coarsening.coarsen(A, levels=coarsening_levels, self_connections=False)
        L = [graph.laplacian(A, normalized=True) for A in graphs]
        model_perf = utils.model_perf()
        '''
        from collections import namedtuple
        Record = namedtuple("gcnn_name", ["gcnn_model", "gcnn_params"])
        ##s = {"test_id1": Record("res1", "time1"), "test_id2": Record("res2", "time2")}
        ##s["test_id1"].resultValue
        model2, gcnn_name2, params2 = self.build_fourier_graph_cnn(Laplacian_list=L)
        model3, gcnn_name3, params3 = self.build_spline_graph_cnn(Laplacian_list=L)
        model4, gcnn_name4, params4 = self.build_chebyshev_graph_cnn(Laplacian_list=L)
        gcnn_model_dicts = {gcnn_name2: Record(model2, params2),
                            gcnn_name3: Record(model3, params3),
                            gcnn_name4: Record(model4, params4)}
        '''
        model2, gcnn_name2 = self.build_fourier_graph_cnn(Laplacian_list=L)
        model3, gcnn_name3 = self.build_spline_graph_cnn(Laplacian_list=L)
        model4, gcnn_name4 = self.build_chebyshev_graph_cnn(Laplacian_list=L)
        gcnn_model_dicts = {gcnn_name2: model2,
                            gcnn_name3: model3,
                            gcnn_name4: model4}
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
            params = self.params[name]
            print(name, params)

            accuracy = [];
            loss = [];
            t_step = [];
            for x_train, y_train, x_val, y_val, tcount in zip(X_train, Y_train, X_val, Y_val, range(len(X_train))):
                train_data = coarsening.perm_data(x_train.reshape(-1, Region_Num), perm)
                train_labels = np.array([d[x] for x in y_train])
                val_data = coarsening.perm_data(x_val.reshape(-1, Region_Num), perm)
                val_labels = np.array([d[x] for x in y_val])
                test_data = coarsening.perm_data(X_test.reshape(-1, Region_Num), perm)
                print('\nFold #%d: training on %d samples with %d features, validating on %d samples and testing on %d samples' %
                      (tcount + 1, train_data.shape[0], train_data.shape[1], val_data.shape[0], test_data.shape[0]))

                acc, los, tstep = model.fit(train_data, train_labels, val_data, val_labels)
                accuracy.append(acc)
                loss.append(los)
                t_step.append(tstep)

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

        return train_acc, test_acc, val_acc