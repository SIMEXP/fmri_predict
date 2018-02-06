Project Updates: 
  website: https://github.com/SIMEXP/fmri_predict/blob/master/linear_model/
  using linear models to predict fMRI activation patterns (motor task-hand movement) using resting-state functional connectivity
  
  The MIST_ROI atlas with 210 regions were used to extract fMRI signals
    1) features: 210*210 region-to-region correlation matrix
    2) output:  210 z-scores from GLM to indicate the probability of brain activation within each region
    3) model: SVR with linear kernel (from sklearn), one seperate model for each region
    4) data: 
      10 sessions of rs-fMRI scans from each subject, dynamic functional connectivity with spliding window size=10mins were used
      10 session of two runs of task-fMRI scans, z-score maps from GLM
    5) training: using both cross-validation (10-fold) and train-test-split from sklearn
    
    6) estimation: either for each region (using MSE) or whole-brain (using correlation)
        for individual region: different models are trained, sometimes lasso/Enet performed better than SVR; mean MSE=0.5
        for whole-brain: correlation between estimated and true activation scores: r=0.3069
