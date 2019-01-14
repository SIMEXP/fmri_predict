#!/home/yuzhang/tensorflow-py3.6/bin/python3.6

# Author: Yu Zhang
# License: simplified BSD
# coding: utf-8
import sys
import os
import warnings
sys.path.append('/home/yuzhang/scratch/HCP/codes/HCP_fmripredict/')

import argparse
from tensorpack.utils import logger
from tensorpack.utils.serialize import dumps, loads

import numpy as np
import importlib
import lmdb
from pathlib import Path
import config, utils
#importlib.reload(utils)

if __name__ == '__main__':
    args = sys.argv[1:]
    logger.set_logger_dir("train_log/svc_simple_log",action="d")
    warnings.simplefilter("ignore")
    #warnings.filterwarnings(action='once')

    parser = argparse.ArgumentParser(description='The description of the parameters')

    parser.add_argument('--task_modality', '-c', help='(required, string) Modality name in Capital for fmri and event design files', type=str)
    parser.add_argument('--subject_to_start', '-f', help='(optional, int,default=0) The index of the first subject in the all_subjects_list for analysis', type=int)
    parser.add_argument('--subject_to_last', '-g', help='(optional, int,default=1086) The index of the last subject in the all_subjects_list for analysis', type=int)
    parser.add_argument('--subjectlist_index', '-l', help='(optional, string, default='') The index indicator of the selected subject list', type=str)

    parser.add_argument('--n_thread', '-t', help='(optional, int, default = 5) Number of threads from each cpu to be used', type=int)
    parser.add_argument('--n_buffersize', '-b', help='(optional, int, default = 50) Number of files to be read at once', type=int)
    parser.add_argument('--n_sessions', '-j', help='(optional, int, default = 0)  Total number of session for the subject', type=int)
    parser.add_argument('--n_sessions_combined', '-x', help='(optional, int, default = 1) The number of sessions to combine', type=int)

    parsed, unknown = parser.parse_known_args(args)

    modality = parsed.task_modality

    startsub = parsed.subject_to_start
    endsub = parsed.subject_to_last
    subjectlist = parsed.subjectlist_index

    n_jobs = 1
    n_thread = parsed.n_thread
    n_buffersize = parsed.n_buffersize
    n_sessions = parsed.n_sessions
    n_sessions_combined = parsed.n_sessions_combined

    #####re-assign parameter settings in config
    config_instance = config.Config()

    if modality:
        config_instance.modality = modality
    if startsub:
        config_instance.startsub = startsub
    if endsub:
        config_instance.endsub = endsub
    if subjectlist:
        config_instance.subjectlist = subjectlist
    if n_thread:
        config_instance.n_thread = n_thread
    if n_buffersize:
        config_instance.n_buffersize = n_buffersize
    if not os.path.exists(config_instance.pathout):
        os.makedirs(config_instance.pathout)

    ###use config parameters to collect fmri data
    '''
    config_instance = config.Config()
    modality = 'MOTOR'
    startsub = 0
    endsub = 2400
    subjectlist = 'ALL'
    '''
    hcp_fmri_instance = utils.hcp_task_fmri(config_instance)

    ##prepare fmri data for analysis
    subjects_trial_label_matrix, sub_name, coding,trial_dura = hcp_fmri_instance.prepare_fmri_files_list()
    print(np.array(subjects_trial_label_matrix).shape)
    print("each trial contains %d volumes/TRs for task %s" % (trial_dura,modality))
    ###updating information in the config settings
    config_instance.task_contrasts = hcp_fmri_instance.task_contrasts
    config_instance.Trial_dura = trial_dura
    config_instance.EVS_files = hcp_fmri_instance.EVS_files
    config_instance.fmri_files = hcp_fmri_instance.fmri_files
    config_instance.confound_files = hcp_fmri_instance.confound_files

    ############
    fmri_files = hcp_fmri_instance.fmri_files
    confound_files = hcp_fmri_instance.confound_files
    print(np.array(subjects_trial_label_matrix).shape)
    #print(np.unique(sub_name), len(sub_name))

    ###output logs
    print("--fmri_folder: ", config_instance.pathfmri)
    print('--temp_out:', config_instance.pathout)
    print('--atlas_filename: %s \n\n' % config_instance.AtlasName)

    mmp_atlas = config_instance.mmp_atlas
    #lmdb_filename = config_instance.pathout+hcp_fmri_instance.modality+'_'+config_instance.AtlasName + '_ROI_act_1200R' + '_test_' + subjectlist + '.lmdb'
    ##subjects_tc_matrix, subname_coding = utils.extract_mean_seris(fmri_files, confound_files, mmp_atlas, lmdb_filename, nr_proc=100, buffer_size=10)
    subjects_tc_matrix, subname_coding = utils.extract_mean_seris_thread(fmri_files, confound_files, mmp_atlas,
                                                                         hcp_fmri_instance.lmdb_filename,
                                                                         hcp_fmri_instance.Trial_Num,
                                                                         nr_thread=config_instance.n_thread, buffer_size=config_instance.n_buffersize)
    print(np.array(subjects_tc_matrix).shape)
    print('\n')

    #####
    sub_name = []
    for ss in subname_coding:
        sub_name.append(ss.split('_')[0])
    hcp_fmri_instance.sub_name = sub_name
    subjects_tc_matrix, subjects_trial_label_matrix = utils.preclean_data_for_shape_match(subjects_tc_matrix,subjects_trial_label_matrix,subname_coding)
    config_instance.Subject_Num = np.array(subjects_tc_matrix).shape[0]
    print(np.array(subjects_trial_label_matrix).shape)
    print(np.array(subjects_tc_matrix).shape)

    '''
    ##only using this for cnn, no need for svm or fc-nn
    ###use config parameters to collect rs-fmri data
    hcp_rsfmri_instance = utils.hcp_rsfmri(config_instance)
    ##prepare fmri data for analysis
    subjects_tc_matrix, mean_corr_matrix = hcp_rsfmri_instance.prepare_rsfmri_files_list(sub_name=sub_name,N_thread=4)

    '''
    print('\n Classify different tasks using simple-svm with rbf kernel...')
    target_name = np.unique(list(hcp_fmri_instance.task_contrasts.values()))
    ##scores= utils.my_svc_simple(subjects_tc_matrix, subjects_trial_label_matrix, target_name, sub_num=1500, block_dura=trial_dura, my_cv_fold=10,my_comp=20)
    ##print(scores)
    '''
    print('\n Changing the validation process by subject-specific split and average within each trial......')
    scores= utils.my_svc_simple_subject_validation_new(subjects_tc_matrix,subjects_trial_label_matrix,target_name,block_dura=trial_dura,my_cv_fold=10,my_testsize=0.2,my_valsize=0.1)
    print(scores)

    print('\n Changing the validation process by subject-specific split...')
    scores= utils.my_svc_simple_subject_validation_new(subjects_tc_matrix,subjects_trial_label_matrix,target_name,block_dura=1,my_cv_fold=10,my_testsize=0.2,my_valsize=0.1)
    print(scores)
    '''
    ##############################
    ####using fully-connected neural networks for classification of fmri tasks
    print('\n Classify different tasks using simple fc-nn...')
    ##utils.build_fc_nn_simple(subjects_tc_matrix, subjects_trial_label_matrix, target_name, layers=5, hidden_size=64,dropout=0.25,batch_size=128)

    print('\n Classify different tasks using simple fc-nn by subject-specific split...')
    utils.build_fc_nn_subject_validation(subjects_tc_matrix,subjects_trial_label_matrix,target_name,block_dura=trial_dura,
                                         layers=5, hidden_size=256,dropout=0.25,batch_size=128,nepochs=50)

    print('\n Classify different tasks using simple fc-nn by subject-specific split and average within each trial...')
    utils.build_fc_nn_subject_validation(subjects_tc_matrix,subjects_trial_label_matrix,target_name,sub_num=100, block_dura=1,
                                         layers=5,hidden_size=256,dropout=0.25,batch_size=128,nepochs=50)


    ###use config parameters to set parameters for graph convolution
    target_name = np.unique(list(hcp_fmri_instance.task_contrasts.values()))
    hcp_gcnn_instance = utils.hcp_gcnn_fmri(config_instance)
    print('\n Classify different tasks using gcn by subject-specific split...')
    train_acc, test_acc, val_acc = hcp_gcnn_instance.build_graph_cnn_subject_validation_new(subjects_tc_matrix, subjects_trial_label_matrix, target_name,block_dura=1,
                                                                                            layers=config_instance.gcnn_layers,hidden_size=config_instance.gcnn_hidden,
                                                                                            pool_size=config_instance.gcnn_pool,batch_size=128, nepochs=50)

    print('\n Classify different tasks using gcn by subject-specific split and average within each trial...')
    ##train_acc_trial, test_acc_trial, val_acc_trial = hcp_gcnn_instance.build_graph_cnn_subject_validation(subjects_tc_matrix, subjects_trial_label_matrix, target_name,block_dura=trial_dura,layers=config_instance.gcnn_layers,hidden_size=config_instance.gcnn_hidden,pool_size=config_instance.gcnn_pool,batch_size=128, nepochs=50)
    train_acc_trial, test_acc_trial, val_acc_trial = hcp_gcnn_instance.build_graph_cnn_subject_validation_new(subjects_tc_matrix, subjects_trial_label_matrix, target_name,block_dura=trial_dura,
                                                                                                              layers=config_instance.gcnn_layers,hidden_size=config_instance.gcnn_hidden,
                                                                                                              pool_size=config_instance.gcnn_pool,batch_size=128, nepochs=50)


    '''
    ####for script testing:
    modality='MOTOR'
    startsub = 0
    endsub = 2400
    subjectlist = 'ALL'

    python ./extract_fmri_event_data.py --task_modality=$modality --subject_to_start=0 --subject_to_last=100 --subjectlist_index='t010'

    '''
