Part I: classify between hand, foot and tongue movements using SVM

1. Multiclass SVM with RBF kernels on activation patterns: 
20 samples (sessions*runs) with 210 features (ROIs)
Results:
  SVM Scoring with 5-fold cross-validation:   mean accuarcy = 1.0
  after PCA decomposition into 20 components: mean accuarcy = 0.916
  after kernel-PCA decomposition into 20 components: mean accuarcy = 0.99
  after ICA decomposition: 			mean accuarcy = 0.75
  after MDS decomposition: 			mean accuarcy = 0.80
[ 0.83333333  0.91666667  0.75        0.75        0.83333333]


2. Multiclass SVM with RBF kernels on fMRI signals:  
1480 samples (sessions*runs*trials) with 210 features (ROIs)
Results:
  SVM Scoring with 10-fold cross-validation:   mean accuarcy = 0.431
  Reduction into 10 components:
	PCA decomposition:	 mean accuarcy = 0.441
	ICA decomposition: 	 mean accuarcy = 0.419
	Kernal-PCA decomposition: mean accuarcy = 0.466
	MDS decomposition: 	 mean accuarcy = 0.415
	ANOVA feature selection based on F-test:  mean accuarcy = 0.438
  Reduction into 20 components:
	PCA decomposition: 	 mean accuarcy = 0.431
	ICA decomposition: 	 mean accuarcy = 0.419
	Kernal-PCA decomposition: mean accuarcy = 0.479
	MDS decomposition: 	 mean accuarcy = 0.415
	ANOVA feature selection based on F-test:  mean accuarcy = 0.433
  Reduction into 50 components:
	PCA decomposition: 	 mean accuarcy = 0.433
	ICA decomposition: 	 mean accuarcy = 0.419
	Kernal-PCA decomposition: mean accuarcy = 0.419
	MDS decomposition: 	 mean accuarcy = 0.452
	ANOVA feature selection based on F-test:  mean accuarcy = 0.439
  Reduction into 100 components:
	PCA decomposition: 	 mean accuarcy = 0.417
	ICA decomposition: 	 mean accuarcy = 0.419
	Kernal-PCA decomposition:  mean accuarcy = 0.419
	MDS decomposition: 	 mean accuarcy = 0.443
	ANOVA feature selection based on F-test:  mean accuarcy = 0.444
  Reduction into 150 components:
	PCA decomposition:  	 mean accuarcy = 0.416
	ICA decomposition:  	 mean accuarcy = 0.419
	Kernal-PCA decomposition:  mean accuarcy = 0.419
	MDS decomposition: 	 mean accuarcy = 0.453
	ANOVA feature selection based on F-test:  mean accuarcy = 0.444


