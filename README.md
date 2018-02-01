# fmri_predict
predicting fmri activaties from connectome

##to work with git
1) git status: to check any changes in the repo
2) git add: to save the changes
3) git commit -a: to save new updates and commit
4) git push: to upload any local changes to github
5) git pull: to clone new changes in github to local computers
6) git log: to check the log information in the repo


##discussion with Pierre on Jan 29th
1) start with a simple model: predicting motor activation from functional connectivity using sparse linear regression model 
2) using atlas: 
      group atlas: MIST with two resolution (200/1000 regions)
      individual atlas
3) defining network structure: 7-functional networks (non-linear relationship could be learned through convolutional layers; thus no logical conflict)
4) for limited training samples: use sliding-windows to generate dynamic functional connectivity (duration:5min)

###Data
1) resting-state: 10 sessions under the folder: /data/cisl/raw_data/midnight/Rawdata/nii_data/preproc_fsl/sub01/rest
using warped_F_sess*_res_ICA_filt_sm6.nii.gz for after ICA-AROMA, temporal filtering, spatial smoothing and registered
2) motor tasks: 10 sessions and 2 runs for each,under the folder: /data/cisl/raw_data/midnight/Rawdata/nii_data/preproc_fsl/sub01/motor
  preprocessed fmri: filtered_func_data_ICA.nii.gz
  brain activation map from contrasts:
  zstat1: foot movement 
  zstat2: hand 
  zstat3: tongue 
  zstat4: foot_left 
  zstat5: foot_right 
  zstat6: hand_left 
  zstat7: hand_right

###first practice: predicting task activation from RSFC using linear model
models: LinearRegression, RidgeRegression, Lasso, ElasticNetCV, LinearSVR 
    for each region, the linear models are trained and the best model are chosen based on cross-validation
data: dynamic functional connectivity (window_size=10min), motor task (2 runs) for 10 sessions
atlas: we used MIST_ROI atlas (210 regions) to extract mean fMRI signal or activation for model training
regions: pre-select regions with moderate activity from the activation maps (z-score>1.9); 
         after that, we trained the linear models for approximately 83 regions, independently

script: midnight_project_resting.ipynb
further considerations: 
      1). combining multi-subject data and using multitask models during training
      2). statistical test on z-maps first and convert the activation map into binary maps. Thus, we could use classification models instead of regression, which might improve prediction accuracy
