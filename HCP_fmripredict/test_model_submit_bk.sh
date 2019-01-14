#!/bin/bash
#SBATCH --account=rrg-pbellec   
#SBATCH --nodes=1      
#SBATCH --tasks-per-node=8   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=200G        # memory per node
#SBATCH --time=0-8:00  #0-12:00      # time (DD-HH:MM)
#SBATCH --output=../train_log/hcp_loaddata_%x_%N-%j.out  # %N for node name, %j for jobID

#module load cuda cudnn python/3.6.3
source activate tensorflow
#mod='WM'
#list='ALL'
mod=$1
list=$2
if [ -z ${mod} ];then mod='MOTOR';fi
if [ -z ${list} ];then list='ALL';fi

##python -W ignore ./extract_fmri_event_data.py --task_modality=${mod} --subject_to_start=0 --subject_to_last=2400 --subjectlist_index=${list} --n_thread=5

python ./extract_fmri_event_data.py --task_modality=${mod} --subject_to_start=0 --subject_to_last=2400 --subjectlist_index=${list} --n_thread=5 --n_buffersize=30

##sbatch --mem=50G --time=0-10:0  --nodes=2 --ntasks-per-node=8 --account=rrg-pbellec --output=../../hcp_loaddata_WM_ALL_logs.txt ./extract_fmri_event_data.py --task_modality='WM' --subject_to_start=0 --subject_to_last=2400 --subjectlist_index='ALL' --n_thread=5
