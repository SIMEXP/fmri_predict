#!/home/yuzhang/jupyter_py3/bin/python

# Author: Yu Zhang
# License: simplified BSD
# coding: utf-8

import sys
sys.path.append('/home/yuzhang/projects/rrg-pbellec/yuzhang/HCP/codes/HCP_fmripredict')

import config, utils

config_instance = config.Config()
print("--modality", config_instance.modality)
print("--fmri_folder: ", config_instance.pathfmri)
print('--temp_out:', config_instance.pathout)
print('--atlas_filename:',config_instance.AtlasName)

hcp_fmri_instance = utils.hcp_task_fmri(config_instance)

