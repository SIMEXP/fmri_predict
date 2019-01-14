#!/home/yuzhang/tensorflow-py3.6/bin/python3.6

# Author: Yu Zhang
# License: simplified BSD
# coding: utf-8

###default parameter settings
class Config():
    pathfmri = '/home/yuzhang/scratch/HCP/aws_s3_HCP1200/FMRI/'
    pathout = '/home/yuzhang/scratch/HCP/temp_res_new/'

    TR = 0.72
    lowcut = 0.01
    highcut = 0.08
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
    pathsource = "/home/yuzhang/scratch/HCP/codes/"
    mmp_atlas = pathsource + "HCP_S1200_GroupAvg_v1/" + "Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii"
    AtlasName = 'MMP'
    Subject_Num = 2400
    Trial_Num = 284
    Node_Num = 32000
    Region_Num = 200

    startsub = 0
    endsub = Subject_Num
    subjectlist = 'ALL'
    n_thread = 5
    n_buffersize = 50

    ##temp saving file
    fmri_filename = 'Atlas.dtseries.nii'
    confound_filename = 'Movement_Regressors.txt'
    rsfmri_filename = 'Atlas_hp2000_clean.dtseries.nii'

    '''
    ###do not update paras in config
    ev_filename = 'event_labels_1200R' + '_test_' + subjectlist + '.h5'  # '.txt'
    fmri_matrix_filename = AtlasName + '_ROI_act_1200R' + '_test_' + subjectlist + '.lmdb' #'.h5' # '.txt'
    #lmdb_filename = config_instance.pathout + hcp_fmri_instance.modality + '_' + fmri_matrix_filename
    '''

    import os
    try:
        ###params for graph_cnn
        import tensorflow as tf
        gcnn = tf.app.flags
        FLAGS = gcnn.FLAGS

        # Graphs.
        gcnn.DEFINE_integer('number_edges', 8, 'Graph: minimum number of edges per vertex.')
        gcnn.DEFINE_bool('normalized_laplacian', True, 'Graph Laplacian: normalized.')
        gcnn.DEFINE_integer('coarsening_levels', 6, 'Number of coarsened graphs.')
        gcnn.DEFINE_string('adj_mat', os.path.join(pathsource, 'MMP_adjacency_mat_white.pconn.nii'), 'Directory to adj matrix on surface data.')
    except ImportError:
        print("Tensorflow is not avaliable in the current node!")

    gcnn_layers = 3
    gcnn_hidden = 256
    gcnn_pool = 4

    gcnn_coarsening_levels = 6
    gcnn_adj_mat_dict = {'surface': os.path.join(pathsource, 'MMP_adjacency_mat_white.pconn.nii'),
                         'SC': os.path.join(pathsource, 'HCP_S1200_GroupAvg_v1/S1200.All.corrThickness_MSMAll.32k_fs_LR.dscalar.nii'),
                         'FC': os.path.join(pathsource, 'HCP_S1200_GroupAvg_v1/S1200.All.corrThickness_MSMAll.32k_fs_LR.dscalar.nii')}
    gcnn_adj_mat_type = 'SC'
