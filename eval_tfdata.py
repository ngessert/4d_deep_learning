import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
import models
import math
from glob import glob
import re
import os
import sys
import importlib
import sklearn.preprocessing
import utils_tfdata
import pickle
import time

# add configuration file
# Dictionary for model configuration
mdlParams = {}

# Import machine config
pc_cfg = importlib.import_module('pc_cfgs.'+sys.argv[1])
mdlParams.update(pc_cfg.mdlParams)

# Always preload
mdlParams['preload'] = True


# This argument controls inference time test
if len(sys.argv) > 5:
    if 'inftimes' in sys.argv[5]:
        mdlParams['inftimes'] = [int(s) for s in re.findall(r'\d+',sys.argv[5])][-1]
    else:
        mdlParams['inftimes'] = 0
else:
    mdlParams['inftimes'] = 0

# Import model config
model_cfg = importlib.import_module('cfgs.'+sys.argv[2])
mdlParams_model = model_cfg.init(mdlParams)
mdlParams.update(mdlParams_model)

# GPU number
if len(sys.argv) > 4:
    if 'gpu' in sys.argv[4]:
       mdlParams['numGPUs'] = [[int(s) for s in re.findall(r'\d+',sys.argv[4])][-1]]

# Path name where model is saved is the fourth argument
if 'NONE' in sys.argv[3]:
    mdlParams['saveDirBase'] = mdlParams['saveDir'] + sys.argv[2]
else:
    mdlParams['saveDirBase'] = sys.argv[3]

# Checkpoint name
if len(sys.argv) > 4:
    if 'best' in sys.argv[4]:
        mdlParams['ckpt_name'] = 'checkpoint_best-'
    else:
        mdlParams['ckpt_name'] = 'checkpoint-'    
else:
    mdlParams['ckpt_name'] = 'checkpoint-'



# Set visible devices
cuda_str = ""
for i in range(len(mdlParams['numGPUs'])):
    cuda_str = cuda_str + str(mdlParams['numGPUs'][i])
    if i is not len(mdlParams['numGPUs'])-1:
        cuda_str = cuda_str + ","
print("Devices to use:",cuda_str)
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_str

# Set training set to eval mode
mdlParams['trainSetState'] = 'eval'

# Check whether this model is part of a CV
#if 'CV' in mdlParams['saveDirBase']:
    # Find out which CV set it is, assumes set is last subfolder
#    cv_num = int(mdlParams['saveDirBase'][-1])
#    mdlParams['valInd'] = mdlParams['valIndCV'][cv_num]
#    mdlParams['trainInd'] = mdlParams['trainIndCV'][cv_num]

# Put all placeholders into one dictionary for feeding
placeholders = {}
# Values to feed during training
feed_list = {}
# Values to feed during testing
feed_list_inference = {}
# Collect model variables
modelVars = {}

# Save results in here
allData = {}

allData['lossBest'] = np.zeros([mdlParams['numCV']])
allData['maeBest'] = np.zeros([mdlParams['numCV'],mdlParams['numOut']])
allData['maestdBest'] = np.zeros([mdlParams['numCV'],mdlParams['numOut']])
allData['rmaeBest'] = np.zeros([mdlParams['numCV'],mdlParams['numOut']])
allData['rmaestdBest'] = np.zeros([mdlParams['numCV'],mdlParams['numOut']])
allData['ACCBest'] = np.zeros([mdlParams['numCV']])
allData['convergeTime'] = np.zeros([mdlParams['numCV']])
allData['bestPred'] = {}
allData['targets'] = {}
# for cv
allData['lossBest_v'] = np.zeros([mdlParams['numCV']])
allData['maeBest_v'] = np.zeros([mdlParams['numCV'],mdlParams['numOut']])
allData['maestdBest_v'] = np.zeros([mdlParams['numCV'],mdlParams['numOut']])
allData['rmaeBest_v'] = np.zeros([mdlParams['numCV'],mdlParams['numOut']])
allData['rmaestdBest_v'] = np.zeros([mdlParams['numCV'],mdlParams['numOut']])
allData['ACCBest_v'] = np.zeros([mdlParams['numCV']])
allData['convergeTime_v'] = np.zeros([mdlParams['numCV']])
allData['bestPred_v'] = {}
allData['targets_v'] = {}
# for train
allData['lossBest_t'] = np.zeros([mdlParams['numCV']])
allData['maeBest_t'] = np.zeros([mdlParams['numCV'],mdlParams['numOut']])
allData['maestdBest_t'] = np.zeros([mdlParams['numCV'],mdlParams['numOut']])
allData['rmaeBest_t'] = np.zeros([mdlParams['numCV'],mdlParams['numOut']])
allData['rmaestdBest_t'] = np.zeros([mdlParams['numCV'],mdlParams['numOut']])
allData['ACCBest_t'] = np.zeros([mdlParams['numCV']])
allData['convergeTime_t'] = np.zeros([mdlParams['numCV']])
allData['bestPred_t'] = {}
allData['targets_t'] = {}

if mdlParams['inftimes'] == 0:
    for cv in range(mdlParams['numCV']):
        # Reset graph
        tf.reset_default_graph()
        # Def current CV set
        mdlParams['trainInd'] = mdlParams['trainIndCV'][cv]
        # Def train eval
        mdlParams['print_trainerr'] = True
        mdlParams['trainInd_eval'] = mdlParams['trainInd']
        if 'valIndCV' in mdlParams:
            mdlParams['valInd'] = mdlParams['valIndCV'][cv]
            print("Valinds",mdlParams['valInd'].shape)
        if 'testIndCV' in mdlParams:
            mdlParams['testInd'] = mdlParams['testIndCV'][cv]
            print("testInds",mdlParams['testInd'].shape)            
        # Def current path for saving stuff
        if 'valIndCV' in mdlParams:
            mdlParams['saveDir'] = mdlParams['saveDirBase'] + '/CVSet' + str(cv)
        else:
            mdlParams['saveDir'] = mdlParams['saveDirBase']

        # Potentially calculate setMean to subtract
        if mdlParams['subtract_set_mean'] == 1:
            mdlParams['setMean'] = np.mean(mdlParams['images_means'][mdlParams['trainInd'],:],(0))
        print("Set Mean",mdlParams['setMean']) 

        # Scaler, scales targets to a range of 0-1
        if mdlParams['scale_targets']:
            mdlParams['scaler'] = sklearn.preprocessing.MinMaxScaler().fit(mdlParams['labels_array'][mdlParams['trainInd'],:][:,mdlParams['outputs'].astype(int)])  
            mdlParams['labels_array_scaled'] = mdlParams['scaler'].transform(mdlParams['labels_array'])

        with tf.device('/cpu:0'):
            # Session config
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            modelVars['Sess'] = tf.Session(config=config)

            # Define queues and model inputs
            # Define base placeholders, that are resued
            placeholders = {}
            placeholders_q_in = {}
            placeholders_q_out = {}
            placeholders['train_state'] = tf.placeholder(tf.bool, name='train_state')
            placeholders['KP1'] = tf.placeholder(tf.float32, name="KP1")
            placeholders['KP2'] = tf.placeholder(tf.float32, name="KP2")

            # Prepare input function function
            dataSetInputFcn = utils_tfdata.getInputFunction(mdlParams,modelVars)    
            # Set up queues
            modelVars['iterator'] = dataSetInputFcn()
            # Set iterator handle to in the placeholders
            placeholders['handle'] = modelVars['handle']

        # Evaluation always with one GPU
        #mdlParams['numGPUs'] = [0]
        with tf.device('gpu:0'):
            modelVars['X_0'], modelVars['Tar_0'], modelVars['Inds_0'] = modelVars['iterator'].get_next()     
            # Multicrop
            print("in",modelVars['X_0'],modelVars['X_0'].get_shape())
            #modelVars['X_0'] = utils_tfdata.image_preprocessing_fn_val_multicrop(modelVars['X_0'], mdlParams['input_size'][0], mdlParams['input_size'][1])  
            #print(modelVars['X_0'].get_shape())           
            # Build graph, put all variables on CPU
            with slim.arg_scope([slim.model_variable, slim.variable], device='/cpu:0'):
                model_function = models.getModel(mdlParams,placeholders)
                modelVars['pred_0'] = model_function(modelVars['X_0'])
            # Build loss
            modelVars['loss_0'] = tf.reduce_mean(tf.square(tf.subtract(modelVars['Tar_0'],modelVars['pred_0'])))
            tf.add_to_collection(tf.GraphKeys.LOSSES, modelVars['loss_0']) 
            # Total loss, in case some regularization is in there
            modelVars['total_loss_0'] = tf.add_n(tf.losses.get_losses(loss_collection=tf.GraphKeys.LOSSES))

        # Value to feed for training/testing
        feed_list_inference['train_state'] = False
        # Set which iterator to use (train/val/test)
        feed_list_inference['handle'] = modelVars['train_eval_handle']
        # Feed values for inference 
        feed_list_inference['KP1'] = 1.0
        feed_list_inference['KP2'] = 1.0

        # Get moving average varibales
        variable_averages = tf.train.ExponentialMovingAverage(mdlParams['moving_avg_var_decay'])
        variables_to_restore = variable_averages.variables_to_restore(slim.get_model_variables()) #slim.get_model_variables() #

        # Get the saver to restore them
        saver = tf.train.Saver(max_to_keep=0,var_list=variables_to_restore)

        # Manually find latest chekcpoint, tf.train.latest_checkpoint is doing weird shit
        files = glob(mdlParams['saveDir']+'/*')
        global_steps = np.zeros([len(files)])
        for i in range(len(files)):
            # Use meta files to find the highest index
            if 'meta' not in files[i]:
                continue
            if mdlParams['ckpt_name'] not in files[i]:
                continue
            # Extract global step
            nums = [int(s) for s in re.findall(r'\d+',files[i])]
            global_steps[i] = nums[-1]
        # Create path with maximum global step found
        chkPath = mdlParams['saveDir'] + '/' + mdlParams['ckpt_name'] + str(int(np.max(global_steps)))
        print("Restoring: ",chkPath)
        # Restore
        saver.restore(modelVars['Sess'], chkPath)
        # Construct pkl filename: config name, last/best, saved epoch number
        pklFileName = sys.argv[2] + "_" + str(int(np.max(global_steps))) + ".pkl"
        # For trainInd
        if mdlParams['print_trainerr']:
            feed_list_inference['handle'] = modelVars['train_eval_handle']
            loss, mae_mean, mae_std, rmae_mean, rmae_std, acc, predictions, targets, allInds = utils_tfdata.getErrForce_mgpu(mdlParams, "trainInd_eval", modelVars, placeholders, feed_list_inference) 
            print("Training Results",cv,":")
            print("Pred size",predictions.shape)
            print("----------------------------------")
            print("Loss",np.mean(loss))
            print("MAE mean",mae_mean,"+-",mae_std)        
            print("rMAE mean",rmae_mean,"+-",rmae_std)
            print("ACC",acc)
            allData['maeBest_t'][cv,:] = mae_mean
            allData['maestdBest_t'][cv,:] = mae_std
            allData['rmaeBest_t'][cv,:] = rmae_mean
            allData['rmaestdBest_t'][cv,:] = rmae_std
            allData['ACCBest_t'][cv] = acc   
            allData['bestPred_t'][cv] = predictions
            allData['targets_t'][cv] = targets            
        # Regression
        if 'valInd' in mdlParams:
            feed_list_inference['handle'] = modelVars['validation_handle']
            loss, mae_mean, mae_std, rmae_mean, rmae_std, acc, predictions, targets, allInds = utils_tfdata.getErrForce_mgpu(mdlParams, "valInd", modelVars, placeholders, feed_list_inference) 
            print("Validation Results",cv,":")
            print("Pred size",predictions.shape)
            print("----------------------------------")
            print("Loss",np.mean(loss))
            print("MAE mean",mae_mean,"+-",mae_std)        
            print("rMAE mean",rmae_mean,"+-",rmae_std)
            print("ACC",acc)
            allData['maeBest_v'][cv,:] = mae_mean
            allData['maestdBest_v'][cv,:] = mae_std
            allData['rmaeBest_v'][cv,:] = rmae_mean
            allData['rmaestdBest_v'][cv,:] = rmae_std
            allData['ACCBest_v'][cv] = acc   
            allData['bestPred_v'][cv] = predictions
            allData['targets_v'][cv] = targets
        if 'testInd' in mdlParams:
            feed_list_inference['handle'] = modelVars['test_handle']
            loss, mae_mean, mae_std, rmae_mean, rmae_std, acc, predictions, targets, allInds = utils_tfdata.getErrForce_mgpu(mdlParams, "testInd", modelVars, placeholders, feed_list_inference) 
            print("Test Results",cv,":")
            print("Pred size",predictions.shape)
            print("----------------------------------")
            print("Loss",np.mean(loss))
            print("MAE mean",mae_mean,"+-",mae_std)        
            print("rMAE mean",rmae_mean,"+-",rmae_std)
            print("ACC",acc)
            allData['maeBest'][cv,:] = mae_mean
            allData['maestdBest'][cv,:] = mae_std
            allData['rmaeBest'][cv,:] = rmae_mean
            allData['rmaestdBest'][cv,:] = rmae_std
            allData['ACCBest'][cv] = acc   
            allData['bestPred'][cv] = predictions
            allData['targets'][cv] = targets                               
    # Mean results over all folds
    print("-------------------------------------------------")
    print("Mean over all Folds")
    print("-------------------------------------------------")
    if mdlParams['print_trainerr']:
        print("trainset")
        print("MAE mean",np.mean(allData['maeBest_t'],0),"+-",np.mean(allData['maestdBest_t'],0))           
        print("rMAE mean",np.mean(allData['rmaeBest_t'],0),"+-",np.mean(allData['rmaestdBest_t'],0))
        print("ACC",np.mean(allData['ACCBest_t']))           
    if 'testInd' in mdlParams:
        print("testset")
        print("MAE mean",np.mean(allData['maeBest'],0),"+-",np.mean(allData['maestdBest'],0))           
        print("rMAE mean",np.mean(allData['rmaeBest'],0),"+-",np.mean(allData['rmaestdBest'],0))
        print("ACC",np.mean(allData['ACCBest']))      
    if 'valInd' in mdlParams:
        print("-------------------------------------------------")
        print("valind")
        print("MAE mean",np.mean(allData['maeBest_v'],0),"+-",np.mean(allData['maestdBest_v'],0))           
        print("rMAE mean",np.mean(allData['rmaeBest_v'],0),"+-",np.mean(allData['rmaestdBest_v'],0))
        print("ACC",np.mean(allData['ACCBest_v']))               
    # Save dict with results
    with open(mdlParams['saveDirBase'] + "/" + pklFileName, 'wb') as f:
        pickle.dump(allData, f, pickle.HIGHEST_PROTOCOL)         

else:    
    print("Saved, now inference times")
    # Inference times
    # Resest graph
    tf.reset_default_graph()
    # Session config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    modelVars['Sess'] = tf.Session(config=config)
    # Just take the first CV model
    mdlParams['saveDir'] = mdlParams['saveDirBase'] + '/CVSet' + str(0)
    # Placeholders
    placeholders = {}
    placeholders['train_state'] = tf.placeholder(tf.bool, name='train_state')
    placeholders['KP1'] = tf.placeholder(tf.float32, name="KP1")
    placeholders['KP2'] = tf.placeholder(tf.float32, name="KP2")

    # Dummy input
    if len(mdlParams['input_size']) == 3:
        placeholders['X'] = tf.placeholder(tf.float32, name="X", shape=[None,mdlParams['timesteps'], mdlParams['input_size'][0],mdlParams['input_size'][1],mdlParams['input_size'][2],1])        
        input_mat = np.zeros([1,mdlParams['timesteps'],mdlParams['input_size'][0],mdlParams['input_size'][1],mdlParams['input_size'][2],1])
    else:
        placeholders['X'] = tf.placeholder(tf.float32, name="X", shape=[None,mdlParams['timesteps'], mdlParams['input_size'][0],mdlParams['input_size'][1],1])        
        input_mat = np.zeros([1,mdlParams['timesteps'],mdlParams['input_size'][0],mdlParams['input_size'][1],1])

    
    mdlParams['batchSize'] = 1
    # Build graph, put all variables on CPU
    with slim.arg_scope([slim.model_variable, slim.variable], device='/cpu:0'):
        model_function = models.getModel(mdlParams,placeholders)
        modelVars['pred'] = model_function(placeholders['X'])

    # Value to feed for training/testing
    feed_list_inference['train_state'] = False
    # Feed dummy
    feed_list_inference['X'] = input_mat
    # Feed values for inference 
    feed_list_inference['KP1'] = 1.0
    feed_list_inference['KP2'] = 1.0

    # Get moving average varibales
    #variable_averages = tf.train.ExponentialMovingAverage(mdlParams['moving_avg_var_decay'])
    variables_to_restore = slim.get_model_variables()#variable_averages.variables_to_restore(slim.get_model_variables()) #slim.get_model_variables()

    # Print number of parameters
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        #print(shape)
        #print(len(shape))
        variable_parameters = 1
        for dim in shape:
            #print(dim)
            variable_parameters *= dim.value
        #print(variable_parameters)
        total_parameters += variable_parameters
    print("Total parameters",total_parameters)

    # Get the saver to restore them
    #saver = tf.train.Saver(max_to_keep=0,var_list=variables_to_restore)

    # Manually find latest chekcpoint, tf.train.latest_checkpoint is doing weird shit
    #files = glob(mdlParams['saveDir']+'/*')
    #global_steps = np.zeros([len(files)])
    #for i in range(len(files)):
    #    # Use meta files to find the highest index
    #    if 'meta' not in files[i]:
    #        continue
    #    if mdlParams['ckpt_name'] not in files[i]:
    #        continue
    #    # Extract global step
    #    nums = [int(s) for s in re.findall(r'\d+',files[i])]
    #    global_steps[i] = nums[-1]
    # Create path with maximum global step found
    #chkPath = mdlParams['saveDir'] + '/' + mdlParams['ckpt_name'] + str(int(np.max(global_steps)))
    #print("Restoring: ",chkPath)
    ## Restore
    #saver.restore(modelVars['Sess'], chkPath)  
    init = tf.global_variables_initializer()
    modelVars['Sess'].run(init)          
    # Get inf times
    inf_times = np.zeros([mdlParams['inftimes']])
    for i in range(mdlParams['inftimes']):
        start_time = time.time()
        _ = modelVars['Sess'].run(modelVars['pred'], feed_dict={placeholders[p]: feed_list_inference[p] for p in placeholders})
        inf_times[i] = time.time()-start_time
    inf_times = np.array(inf_times)
    print("Inference times for",mdlParams['inftimes'],"repetitions",np.mean(inf_times[10:]),"+-",np.std(inf_times[10:]))
    print(inf_times)