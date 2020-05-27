import os
import sys
import h5py
import numpy as np
from glob import glob

# Write to file instead
#sys.stdout = open(os.environ['WORK']+'/log/test_u1.txt','w')
#sys.stderr = open(os.environ['WORK']+'/log/err_u1.txt','w')

def init(mdlParams_):
    mdlParams = {}
    # Save summaries and model here
    mdlParams['saveDir'] = mdlParams_['pathBase']+'/data/omesforce/'
    # Data is loaded from here
    mdlParams['dataDir'] = mdlParams_['pathBase']+'/data/omesforce/4d'
    # Number of GPUs on device
    mdlParams['numGPUs'] = [0]

    ### Model Selection ###
    mdlParams['model_type'] = 'Resnet3D'
    #mdlParams['batch_func_type'] = 'getBatchOpto'
    mdlParams['input_func_type'] = 'force' 
    mdlParams['h5datanames'] = ['mips','forces_t_mean','indices_0']
    # All sets to be considered
    mdlParams['dataset_names'] = ['Force4D_3__1','Force4D_3__2','Force4D_3__3','Force4D_3__4','Force4D_3__5','Force4D_3__6','Force4D_3__7','Force4D_3__8',
                                  'Force4D_3__9','Force4D_3__10','Force4D_3__11','Force4D_3__12','Force4D_3__13','Force4D_3__14','Force4D_3__15','Force4D_3__16',
                                  'Force4D_3__17','Force4D_3__18','Force4D_3__19','Force4D_3__20']
    # Subset for validation, if no mix
    mdlParams['dataset_names_val'] = [['Force4D_3__3','Force4D_3__7']]
    mdlParams['dataset_names_test'] = ['Force4D_3__5','Force4D_3__15']
    mdlParams['numCV'] = 1
    # CV, if no fixed val set
    mdlParams['useCV'] = True
    # Potentially subtract a mean
    mdlParams['setMean'] = 0   
    # Number of outputs to consider
    mdlParams['outputs'] = np.array([0])
    mdlParams['numOut'] = mdlParams['outputs'].shape[0]
    # Number of channels, always 1 for OCT
    mdlParams['numChannels'] = 1
    # Per image standardization
    mdlParams['standardize'] = False
    # Scale to 0-1
    mdlParams['normalize'] = True    
    # Save predictions from time to time
    mdlParams['save_pred_freq'] = 10000    
    # Number of time steps
    mdlParams['timesteps'] = 6
    mdlParams['target_shift'] = 0

    ### 3D CNN ResNetA Parameters ###
    # Parameters for the initial convolution
    mdlParams['strides_init'] = [1,2,2]
    mdlParams['num_filters_init'] = 16
    mdlParams['kernel_init'] = [5,5,5]
    mdlParams['factorize'] = False
    # Resnet architecture definition
    mdlParams['ResNet_Size'] = [2,3,4]
    # Strides
    mdlParams['ResNet_Stride'] = [[1,2,2],[1,2,2],[1,2,2]]
    # Feature map size
    mdlParams['ResNet_FM'] = [16,32,64]

    ### Training Parameters ###
    # Batch size
    mdlParams['batchSize'] = 8
    # Initial learning rate
    mdlParams['learning_rate'] = 0.00005
    # Lower learning rate after no improvement over 100 epochs
    mdlParams['lowerLRAfter'] = 50
    # If there is no validation set, start lowering the LR after X steps
    mdlParams['lowerLRat'] = 100
    # Divide learning rate by this value
    mdlParams['LRstep'] = 2
    # Maximum number of training iterations
    mdlParams['training_steps'] = 100
    # Display error every X steps
    mdlParams['display_step'] = 5
    # Scale?
    mdlParams['scale_targets'] = False
    # Peak at test error during training? (generally, dont do this!)
    mdlParams['peak_at_testerr'] = False
    # Print trainerr
    mdlParams['print_trainerr'] = False
    # Decay of moving averages
    mdlParams['moving_avg_var_decay'] = 0.999
    # Potential dropout 1
    mdlParams['keep_prob_1'] = 1.0
    # Potential dropout 2
    mdlParams['keep_prob_2'] = 1.0
    # Subtract trainset mean?
    mdlParams['subtract_set_mean'] = False
    # Preload?
    mdlParams['preload'] = True

    ### Data Augmentation Options ###
    # Flip along x direction
    mdlParams['flip_x'] = False
    # Flip along y direction
    mdlParams['flip_y'] = False
    # Random lateral 90° rotations
    mdlParams['rot_lat'] = False

    ### Data ###
    # Define dicts
    mdlParams['data_paths'] = []
    mdlParams['input_size'] = [32,32]
    # First: get all paths into dict
    # All sets
    allSets = sorted(glob(mdlParams['dataDir']  + '/*'))    
    # Indice set
    mdlParams['trainIndCV'] = []
    mdlParams['valIndCV'] = []
    # Extract setids
    for i in range(len(allSets)):
        # Check if want to include this dataset
        foundSet = False
        for j in range(len(mdlParams['dataset_names'])):
            if mdlParams['dataset_names'][j] in allSets[i]:
                foundSet = True
                break
        if not foundSet:
            continue       
        # Write file names into list
        mdlParams['data_paths'].append(allSets[i])
    if mdlParams['preload']:
        for k in range(mdlParams['numCV']):
            mdlParams['trainIndCV'].append(np.array([]))
            mdlParams['valIndCV'].append(np.array([]))   
        mdlParams['testInd'] = np.array([])     
        # Create labels and data volume        
        for i in range(len(mdlParams['data_paths'])):
            print("Curr File",mdlParams['data_paths'][i])
            with h5py.File(mdlParams['data_paths'][i]) as vols:
                # Append and define valid indices
                if i == 0:
                    mdlParams['data_array'] = np.transpose(vols[mdlParams['h5datanames'][0]].value)
                    # Expand
                    if len(mdlParams['data_array'].shape) == 3:
                        mdlParams['data_array'] = np.expand_dims(mdlParams['data_array'],axis=3)
                    mdlParams['labels_array'] = np.transpose(vols[mdlParams['h5datanames'][1]].value)
                    # Apply indices
                    indices = np.squeeze(np.transpose(vols[mdlParams['h5datanames'][2]].value).astype(int)-1)
                    print("Index shape",indices.shape)
                    mdlParams['labels_array'] = mdlParams['labels_array'][indices,:]
                    # Add indices to respective index set
                    for k in range(mdlParams['numCV']):
                        found = False
                        for set in mdlParams['dataset_names_val'][k]:
                            if set in mdlParams['data_paths'][i]:
                                mdlParams['valIndCV'][k] = np.concatenate((mdlParams['valIndCV'][k],np.array(np.arange(mdlParams['timesteps'],mdlParams['labels_array'].shape[0]))))
                                found = True
                                break     
                        for set in mdlParams['dataset_names_test']:
                            if set in mdlParams['data_paths'][i]:
                                mdlParams['testInd'] = np.concatenate((mdlParams['testInd'],np.array(np.arange(mdlParams['timesteps'],mdlParams['labels_array'].shape[0]))))
                                found = True
                                break                               
                        if not found:
                            mdlParams['trainIndCV'][k] = np.concatenate((mdlParams['trainIndCV'][k],np.array(np.arange(mdlParams['timesteps'],mdlParams['labels_array'].shape[0]))))
                    mdlParams['valid_indices'] = np.array(np.arange(mdlParams['timesteps'],mdlParams['labels_array'].shape[0]))
                else:
                    new_vols = np.transpose(vols[mdlParams['h5datanames'][0]].value)
                    # Expand
                    if len(new_vols.shape) == 3:
                        new_vols = np.expand_dims(new_vols,axis=3)                    
                    new_tars = np.transpose(vols[mdlParams['h5datanames'][1]].value)
                    # Apply indices
                    indices = np.squeeze(np.transpose(vols[mdlParams['h5datanames'][2]].value).astype(int)-1)
                    new_tars = new_tars[indices,:]  
                    # Concat                  
                    mdlParams['data_array'] = np.concatenate((mdlParams['data_array'],new_vols),axis=0)
                    mdlParams['labels_array'] = np.concatenate((mdlParams['labels_array'],new_tars),axis=0)
                    for k in range(mdlParams['numCV']):
                        found = False
                        for set in mdlParams['dataset_names_val'][k]:
                            if set in mdlParams['data_paths'][i]:
                                mdlParams['valIndCV'][k] = np.concatenate((mdlParams['valIndCV'][k],np.array(np.arange(mdlParams['valid_indices'][-1]+mdlParams['timesteps'],mdlParams['valid_indices'][-1]+new_tars.shape[0]+1))))
                                found = True
                                break 
                        for set in mdlParams['dataset_names_test']:
                            if set in mdlParams['data_paths'][i]:
                                mdlParams['testInd'] = np.concatenate((mdlParams['testInd'],np.array(np.arange(mdlParams['valid_indices'][-1]+mdlParams['timesteps'],mdlParams['valid_indices'][-1]+new_tars.shape[0]+1))))
                                found = True
                                break                                   
                        if not found:
                            mdlParams['trainIndCV'][k] = np.concatenate((mdlParams['trainIndCV'][k],np.array(np.arange(mdlParams['valid_indices'][-1]+mdlParams['timesteps'],mdlParams['valid_indices'][-1]+new_tars.shape[0]+1))))                    
                    mdlParams['valid_indices'] = np.concatenate((mdlParams['valid_indices'],np.array(np.arange(mdlParams['valid_indices'][-1]+mdlParams['timesteps'],mdlParams['valid_indices'][-1]+new_tars.shape[0]+1))))
        # Expand
        if len(mdlParams['data_array'].shape) == 3:
            mdlParams['data_array'] = np.expand_dims(mdlParams['data_array'],axis=3)
        # Standardize
        if mdlParams['standardize']:
            for i in range(len(mdlParams['data_array'])):
                for j in range(mdlParams['data_array'].shape[-1]):
                    im_mean = np.mean(mdlParams['data_array'][i,:,:,j])
                    im_std = np.std(mdlParams['data_array'][i,:,:,j])
                    mdlParams['data_array'][i,:,:,j] = (mdlParams['data_array'][i,:,:,j] - im_mean) / (np.sqrt(im_std + 1e-08)) 
        mdlParams['data_array'] = mdlParams['data_array'].astype(np.float32)  
        print("Data type",mdlParams['data_array'].dtype)     
        if mdlParams['normalize']:      
            for i in range(len(mdlParams['data_array'])):
                for j in range(mdlParams['data_array'].shape[-1]):
                    # Normalize to pixelrange
                    im_min = 1
                    im_max = 615
                    mdlParams['data_array'][i,:,:,j] = (mdlParams['data_array'][i,:,:,j] - im_min) / (im_max-im_min)              
        print("Data list",mdlParams['data_array'].shape)  
        print("Tar list",mdlParams['labels_array'].shape)          
    else:
        # TODO: implement
        print("Preload only")        
    # Ind properties
    print("Train")
    for i in range(len(mdlParams['trainIndCV'])):
        mdlParams['trainIndCV'][i] = mdlParams['trainIndCV'][i].astype(int)
        print(mdlParams['trainIndCV'][i].shape)
        print("min",np.min(mdlParams['trainIndCV'][i]),"max",np.max(mdlParams['trainIndCV'][i]))
    print("Val")
    for i in range(len(mdlParams['valIndCV'])):
        mdlParams['valIndCV'][i] = mdlParams['valIndCV'][i].astype(int)
        print(mdlParams['valIndCV'][i].shape)      
        print("min",np.min(mdlParams['valIndCV'][i]),"max",np.max(mdlParams['valIndCV'][i]))
        print("Intersect",np.intersect1d(mdlParams['trainIndCV'][i],mdlParams['valIndCV'][i]))
    if 'testInd' in mdlParams:
        print("Test")
        mdlParams['testInd'] = mdlParams['testInd'].astype(int)
        print(mdlParams['testInd'].shape)      
        print("min",np.min(mdlParams['testInd']),"max",np.max(mdlParams['testInd']))
        for i in range(len(mdlParams['trainIndCV'])):
            print("Intersect test",np.intersect1d(mdlParams['trainIndCV'][i],mdlParams['testInd']))            
    return mdlParams