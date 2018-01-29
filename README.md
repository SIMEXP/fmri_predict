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
