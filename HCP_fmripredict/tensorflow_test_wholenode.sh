#!/bin/bash
#SBATCH --account=def-pbellec 
#SBATCH --job-name=cnn_graph
#SBATCH --nodes=1 
#SBATCH --gres=gpu:lgpu:4  ##request whole node   
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24    # There are 24 CPU cores on Cedar GPU nodes
#SBATCH --mem=0               # Request the full memory of the node
#SBATCH --time=00-15:00      # time (DD-HH:MM)

#SBATCH --output=/home/yuzhang/scratch/HCP/codes/train_log/hcp_task_classify_%x_%N-%j.out ###--output=%N-%j.out  # %N for node name, %j for jobID
####SBATCH --error=/home/yuzhang/scratch/HCP/codes/train_log/hcp_task_classify_%x_%N-%j_%A_%a.err
#SBATCH --workdir="/home/yuzhang/scratch/HCP/codes/HCP_fmripredict/"

module load cuda cudnn python/3.6.3
source $HOME/tensorflow-py3.6/bin/activate
ps | grep python; pkill python;

#python ./tensorflow-test.py

mod=$1
list=$2
if [ -z ${mod} ];then mod='WM';fi
if [ -z ${list} ];then list='ALL';fi

###python -W ignore ./extract_fmri_event_data.py --task_modality=${mod} --subject_to_start=0 --subject_to_last=2400 --subjectlist_index=${list} --n_thread=5

##python -W ignore ./HCP_task_fmri_cnn_tensorpack.py 
python ./HCP_task_fmri_cnn_tensorpack_changesize_bk4_wm.py
