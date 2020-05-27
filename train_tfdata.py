import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import clip_ops
import tensorflow.contrib.slim as slim
import numpy as np
import models
import threading
import pickle
from pathlib import Path
import math
import os
import sys
from glob import glob
import re
import h5py
import importlib
import time
import sklearn.preprocessing
import utils_tfdata
from sklearn.utils import class_weight
#from tensorflow.python.client import device_lib

#device_lib.list_local_devices()

# add configuration file
# Dictionary for model configuration
mdlParams = {}

# Import machine config
pc_cfg = importlib.import_module('pc_cfgs.'+sys.argv[1])
mdlParams.update(pc_cfg.mdlParams)
mdlParams['pc_cfg'] = sys.argv[1]

# Preload?
if len(sys.argv) > 4: 
    if 'preload' in sys.argv[4]:
        mdlParams['preload'] = True   
        # Suppres info
        #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
        #tf.logging.set_verbosity(tf.logging.ERROR)       
    else:
       mdlParams['preload'] = False   
print("Preload ",mdlParams['preload'])

# Selected CV sets?
if len(sys.argv) > 5:
    if 'cv' in sys.argv[5]:
        mdlParams['all_cv_sets'] = [int(s) for s in re.findall(r'\d+',sys.argv[5])]

# Import model config
model_cfg = importlib.import_module('cfgs.'+sys.argv[2])
mdlParams_model = model_cfg.init(mdlParams)
mdlParams.update(mdlParams_model)

# GPU number
if len(sys.argv) > 3:
    if 'gpu' in sys.argv[3]:
       mdlParams['numGPUs'] = [[int(s) for s in re.findall(r'\d+',sys.argv[3])][-1]]

# Set visible devices
cuda_str = ""
for i in range(len(mdlParams['numGPUs'])):
    cuda_str = cuda_str + str(mdlParams['numGPUs'][i])
    if i is not len(mdlParams['numGPUs'])-1:
        cuda_str = cuda_str + ","
print("Devices to use:",cuda_str)
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_str

# Indicate training
mdlParams['trainSetState'] = 'train'

# Path name from filename
mdlParams['saveDirBase'] = mdlParams['saveDir'] + sys.argv[2]

# Check if there is a validation set, if not, evaluate train error instead
if 'valIndCV' in mdlParams or 'valInd' in mdlParams:
    eval_set = 'valInd'
    print("Evaluating on validation set during training.")
else:
    eval_set = 'trainInd'
    print("No validation set, evaluating on training set during training.")


# Put all placeholders into one dictionary for feeding
placeholders = {}
# Values to feed during training
feed_list = {}
# Values to feed during testing
feed_list_inference = {}
# Collect model variables
modelVars = {}

# Check if there were previous ones that have alreary been learned
prevFile = Path(mdlParams['saveDirBase'] + '/CV.pkl')
#print(prevFile)
if prevFile.exists():
    print("Part of CV already done")
    with open(mdlParams['saveDirBase'] + '/CV.pkl', 'rb') as f:
        allData = pickle.load(f)
else:
    allData = {}
    allData['loss_all'] = {}
    allData['lossBest'] = {}
    allData['maeBest'] = {}
    allData['maeStdBest'] = {}
    allData['rmaeBest'] = {}
    allData['rmaeStdBest'] = {}
    allData['accBest'] = {}
    allData['convergeTime'] = {}
    allData['bestPred'] = {}
    allData['targets'] = {}

# Compatibility
if 'all_cv_sets' not in mdlParams:
    mdlParams['all_cv_sets'] = range(mdlParams['numCV'])  
else:
    print("Training on CV sets:",mdlParams['all_cv_sets'])  

# Take care of CV
for cv in mdlParams['all_cv_sets']:
    # Check if this fold was already trained
    if cv in allData['maeBest']:
        print('Fold ' + str(cv) + ' already trained.')
        continue          
    # Reset graph
    tf.reset_default_graph()
    # Def current CV set
    mdlParams['trainInd'] = mdlParams['trainIndCV'][cv]
    # For potential train err eval
    mdlParams['trainInd_eval'] = mdlParams['trainInd']
    if 'valIndCV' in mdlParams:
        mdlParams['valInd'] = mdlParams['valIndCV'][cv]
    if 'testIndCV' in mdlParams:
        mdlParams['testInd'] = mdlParams['testIndCV'][cv]
    # Def current path for saving stuff
    if 'valIndCV' in mdlParams:
        mdlParams['saveDir'] = mdlParams['saveDirBase'] + '/CVSet' + str(cv)
    else:
        mdlParams['saveDir'] = mdlParams['saveDirBase']
    # Create basepath if it doesnt exist yet
    if not os.path.isdir(mdlParams['saveDirBase']):
        os.mkdir(mdlParams['saveDirBase'])
    # Check if there is something to load
    load_old = 0
    if os.path.isdir(mdlParams['saveDir']):
        # Check if a real checkpoint is in there (more than 4 files)
        if len([name for name in os.listdir(mdlParams['saveDir'])]) > 4:
            load_old = 1
            print("Loading old model")
        else:
            # Delete whatever is in there
            filelist = [os.remove(mdlParams['saveDir'] +'/'+f) for f in os.listdir(mdlParams['saveDir'])]
    else:
        os.mkdir(mdlParams['saveDir'])

    if mdlParams['scale_targets']:
        mdlParams['scaler'] = sklearn.preprocessing.MinMaxScaler().fit(mdlParams['labels_array'][mdlParams['trainInd'],mdlParams['outputs'].astype(int)])  
        mdlParams['labels_array'] = mdlParams['scaler'].transform(mdlParams['labels_array'])        

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
        # Tower gradients for each gpu
        tower_grads = []
        # Set up otimizer
        # Changeable learning rate
        lr = tf.get_variable("learning_rate", [],trainable=False,initializer=init_ops.constant_initializer(mdlParams['learning_rate']))
        # Def global step
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(lr,0.9,0.999,0.000001)                                               
        # Summaries
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summaries.append(tf.summary.scalar("learning_rate/learning_rate",lr))
        #modelVars['X_0'], modelVars['Tar_0'] = modelVars['iterator'].get_next()
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(len(mdlParams['numGPUs'])):
            with tf.device('gpu:%d'%mdlParams['numGPUs'][i]):
                with tf.name_scope('tower_%d'%i) as scope:
                    # Batches for tower
                    modelVars['X_'+str(i)], modelVars['Tar_'+str(i)], modelVars['Inds_'+str(i)] = modelVars['iterator'].get_next()
                    #summaries.append(tf.summary.image('model input',modelVars['X_'+str(i)][:,:,:,0:],max_outputs=5))
                    print("Input",modelVars['X_'+str(i)],modelVars['X_'+str(i)].get_shape(),"Output",modelVars['Tar_'+str(i)],modelVars['Tar_'+str(i)].get_shape())                            
                    # Build graph, put all variables on CPU
                    with slim.arg_scope([slim.model_variable, slim.variable], device='/cpu:0'):
                        model_function = models.getModel(mdlParams,placeholders)
                        modelVars['pred_'+str(i)] = model_function(modelVars['X_'+str(i)])
                    print("Pred",modelVars['pred_'+str(i)],modelVars['pred_'+str(i)].get_shape())
                    # Build loss
                    #modelVars['loss_'+str(i)] = tf.losses.mean_squared_error(labels=modelVars['Tar_'+str(i)], predictions=modelVars['pred_'+str(i)], scope=scope, loss_collection=tf.GraphKeys.LOSSES)    
                    modelVars['loss_'+str(i)] = tf.reduce_mean(tf.square(tf.subtract(modelVars['Tar_'+str(i)],modelVars['pred_'+str(i)])))
                    tf.add_to_collection(tf.GraphKeys.LOSSES, modelVars['loss_'+str(i)])                         
                    # Reuse
                    tf.get_variable_scope().reuse_variables()                        
                    # Get relevant variables for training
                    vars_for_training = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                    # Total loss, in case some regularization is in there
                    modelVars['total_loss_'+str(i)] = tf.add_n(tf.losses.get_losses(scope=scope,loss_collection=tf.GraphKeys.LOSSES))
                    # Compute gradients
                    grads = optimizer.compute_gradients(modelVars['total_loss_'+str(i)],var_list=vars_for_training)
                    # Append
                    tower_grads.append(grads)
    # Collect gradients, define updates
    with tf.device('/cpu:0'):
        with tf.name_scope('average_gradient_over_towers'):
            grads = utils_tfdata.average_gradients(tower_grads)
            #global_grad_norm = clip_ops.global_norm(list(zip(*grads))[0])
            #summaries.append(tf.summary.scalar("global_norm/gradient_norm",global_grad_norm))
            # And for all gradients
            #for grad, var in grads:
            #    if grad is not None:
            #        #summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
            #        summaries.append(tf.summary.scalar(var.op.name + '/gradient_norm',clip_ops.global_norm([grad])))
        # Now update
        # Moving average variables
        moving_average_variables = slim.get_model_variables()
        variable_averages = tf.train.ExponentialMovingAverage(mdlParams['moving_avg_var_decay'], global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops.append(variable_averages.apply(moving_average_variables))
        with tf.control_dependencies(update_ops):
            modelVars['Train'] = optimizer.apply_gradients(grads, global_step = global_step,name='train_op_mgpu')


        # Value to feed for training/testing
        feed_list['train_state'] = True
        feed_list_inference['train_state'] = False
        # Set which iterator to use (train/val/test)
        feed_list['handle'] = modelVars['training_handle']
        feed_list_inference['handle'] = modelVars['training_handle']
        # Keep prob values to feed for training
        feed_list['KP1'] = mdlParams['keep_prob_1']
        feed_list['KP2'] = mdlParams['keep_prob_2']
        # Feed values for inference 
        feed_list_inference['KP1'] = 1.0
        feed_list_inference['KP2'] = 1.0

        # Add variable summaries
        #for var in slim.get_model_variables():
        #    summaries.append(tf.summary.histogram(var.op.name, var))

        # Merge summaries
        all_summaries = tf.summary.merge(summaries)

        # Saver
        saver = tf.train.Saver(max_to_keep=0)

        # Initialize the variables (i.e. assign their default value)
        if load_old:
            # Manually find latest chekcpoint, tf.train.latest_checkpoint is doing weird shit
            files = glob(mdlParams['saveDir']+'/*')
            global_steps = np.zeros([len(files)])
            for i in range(len(files)):
                # Use meta files to find the highest index
                if 'meta' not in files[i]:
                    continue
                if 'checkpoint-' not in files[i]:
                    continue                
                # Extract global step
                nums = [int(s) for s in re.findall(r'\d+',files[i])]
                global_steps[i] = nums[-1]
            # Create path with maximum global step found
            chkPath = mdlParams['saveDir'] + '/checkpoint-' + str(int(np.max(global_steps)))
            print("Restoring: ",chkPath)
            # Restore
            saver.restore(modelVars['Sess'], chkPath)
            start_epoch = int(np.max(global_steps))+1
            mdlParams['lastLRUpdate'] = start_epoch
            mdlParams['total_iterations'] = mdlParams.get('warm_up',0)
            # Set last best ind to current
            mdlParams['lastBestInd'] = start_epoch
        else:
            init = tf.global_variables_initializer()
            modelVars['Sess'].run(init)
            start_epoch = 1
            mdlParams['total_iterations'] = 0
            mdlParams['lastLRUpdate'] = 0
            mdlParams['lastBestInd'] = -1


        # Num batches
        numBatchesTrain = int(math.ceil(len(mdlParams['trainInd'])/mdlParams['batchSize']/len(mdlParams['numGPUs'])))
        print("Train batches",numBatchesTrain)

        # Track metrics for lowering LR
        mdlParams['valBest'] = 1000


        # Start training       
        # Initialize summaries writer
        sum_writer = tf.summary.FileWriter(mdlParams['saveDir'], modelVars['Sess'].graph)
        # Run training
        start_time = time.time()
        print("Start training...")
    for step in range(start_epoch, mdlParams['training_steps']+1):
        for j in range(numBatchesTrain):
            mdlParams['total_iterations'] += 1      
            if mdlParams.get('warm_up',0) > 0 and mdlParams['total_iterations'] <= mdlParams.get('warm_up',0):
                update_op = lr.assign(mdlParams['learning_rate']*(mdlParams['total_iterations']/mdlParams.get('warm_up',0)))
                modelVars['Sess'].run(update_op)                       
            # Run optimization op (backprop)
            if len(mdlParams['numGPUs']) > 1:
                #t1 = time.time()
                _, indlen = modelVars['Sess'].run([modelVars['Train'], modelVars['Inds_0']], feed_dict={placeholders[p]: feed_list[p] for p in placeholders})
                print("indlen",indlen)
                #print(np.setdiff1d(inds,mdlParams['class_indices'][6]))
                #print("Mainloop",j," ",time.time()-t1)
            else:
                _, indlen = modelVars['Sess'].run([modelVars['Train'], modelVars['Inds_0']], feed_dict={placeholders[p]: feed_list[p] for p in placeholders})
           
        # Print loss after every peoch
        #print("Current Batch loss",batch_loss)
        if step % mdlParams['display_step'] == 0 or step == 1:
            # Duration so far
            duration = time.time() - start_time
            print("Time",duration)
            # Update summaries
            summary_str = modelVars['Sess'].run(all_summaries, feed_dict={placeholders[p]: feed_list_inference[p] for p in placeholders})
            sum_writer.add_summary(summary_str, step)
            feed_list_inference['handle'] = modelVars['validation_handle']
            # Calculate evaluation metrics
            loss, mae_mean, mae_std, rmae_mean, rmae_std, acc, predictions, targets, allInds = utils_tfdata.getErrForce_mgpu(mdlParams, eval_set, modelVars, placeholders, feed_list_inference)
            # Track all metrics as summaries
            sum_acc = tf.Summary(value=[tf.Summary.Value(tag='MAE', simple_value = np.mean(mae_mean))])
            sum_writer.add_summary(sum_acc, step)
            sum_acc = tf.Summary(value=[tf.Summary.Value(tag='Loss', simple_value = np.mean(loss))])
            sum_writer.add_summary(sum_acc, step)            
            sum_sens = tf.Summary(value=[tf.Summary.Value(tag='MAE_STD', simple_value = np.mean(mae_std))])
            sum_writer.add_summary(sum_sens, step)
            sum_spec = tf.Summary(value=[tf.Summary.Value(tag='RMAE', simple_value = np.mean(rmae_mean))])
            sum_writer.add_summary(sum_spec, step)
            sum_loss = tf.Summary(value=[tf.Summary.Value(tag='RMAE_STD', simple_value = np.mean(rmae_std))])
            sum_writer.add_summary(sum_loss, step) 
            sum_loss = tf.Summary(value=[tf.Summary.Value(tag='ACC', simple_value = np.mean(acc))])
            sum_writer.add_summary(sum_loss, step)                                             
            # Used for early stopping
            eval_metric = np.mean(mae_mean)                                                    
            # Check if we have a new best value
            if eval_metric < mdlParams['valBest']:
                mdlParams['valBest'] = eval_metric            
                allData['lossBest'][cv] = loss
                allData['maeBest'][cv] = mae_mean
                allData['maeStdBest'][cv] = mae_std
                allData['rmaeBest'][cv] = rmae_mean
                allData['rmaeStdBest'][cv] = rmae_std
                allData['accBest'][cv] = acc
                oldBestInd = mdlParams['lastBestInd']
                mdlParams['lastBestInd'] = step
                allData['convergeTime'][cv] = step
                # Save best predictions
                allData['bestPred'][cv] = predictions
                allData['targets'][cv] = targets
                # Delte previously best model
                if os.path.isfile(mdlParams['saveDir'] + '/checkpoint_best-' + str(oldBestInd) + '.index'):
                    os.remove(mdlParams['saveDir'] + '/checkpoint_best-' + str(oldBestInd) + '.data-00000-of-00001')
                    os.remove(mdlParams['saveDir'] + '/checkpoint_best-' + str(oldBestInd) + '.index')
                    os.remove(mdlParams['saveDir'] + '/checkpoint_best-' + str(oldBestInd) + '.meta')
                # Save currently best model
                saver.save(modelVars['Sess'], mdlParams['saveDir'] + '/checkpoint_best', global_step=step)
            # If its not better, just save it delete the last checkpoint if it is not current best one
            # Save current model
            saver.save(modelVars['Sess'], mdlParams['saveDir'] + '/checkpoint', global_step=step)                
            # Delete last one
            if step == mdlParams['display_step']:
                lastInd = 1
            else:
                lastInd = step-mdlParams['display_step']
            if os.path.isfile(mdlParams['saveDir'] + '/checkpoint-' + str(lastInd) + '.index'):
                os.remove(mdlParams['saveDir'] + '/checkpoint-' + str(lastInd) + '.data-00000-of-00001')
                os.remove(mdlParams['saveDir'] + '/checkpoint-' + str(lastInd) + '.index')
                os.remove(mdlParams['saveDir'] + '/checkpoint-' + str(lastInd) + '.meta')                    
            # Potentially save a prediction for train progression investigation
            if step % mdlParams['save_pred_freq'] == 0:
                with h5py.File(mdlParams['saveDir'] + '/pred_epoch_' + str(step) + '.h5') as f:
                    f.create_dataset('pred',data=predictions[0], compression="gzip", compression_opts=9)
                    f.create_dataset('tar',data=targets[0], compression="gzip", compression_opts=9)
            # Print
            print("\n")
            print("CFG:",sys.argv[2])
            print('Fold: %d Epoch: %d/%d (%d h %d m %d s)' % (cv,step,mdlParams['training_steps'], int(duration/3600), int(np.mod(duration,3600)/60), int(np.mod(np.mod(duration,3600),60))) + time.strftime("%d.%m.-%H:%M:%S", time.localtime()))
            print("Loss on ",eval_set,"set: ",loss," MAE: ",mae_mean,"+-",mae_std ," rMAE: ",rmae_mean,"+-",rmae_std," (best MAE: ",mdlParams['valBest']," at Epoch ",mdlParams['lastBestInd'],")")
            print("ACC",acc)             
            # Potentially peek at test error
            if mdlParams['peak_at_testerr']:
                # Adjust handle
                feed_list_inference['handle'] = modelVars['test_handle']                
                loss, mae_mean, mae_std, rmae_mean, rmae_std, acc, predictions, targets, allInds = utils_tfdata.getErrForce_mgpu(mdlParams, "testInd", modelVars, placeholders, feed_list_inference)
                print("\n")
                print("Loss on ","testInd","set: ",loss," MAE: ",mae_mean,"+-",mae_std ," rMAE: ",rmae_mean,"+-",rmae_std)
                print("ACC",acc)  
            # Potentially print train err
            if mdlParams['print_trainerr'] and 'train' not in eval_set:
                # Adjust handle
                feed_list_inference['handle'] = modelVars['train_eval_handle']                
                loss, mae_mean, mae_std, rmae_mean, rmae_std, acc, predictions, targets, allInds = utils_tfdata.getErrForce_mgpu(mdlParams, "trainInd_eval", modelVars, placeholders, feed_list_inference)
                print("\n")
                print("Loss on ","trainInd","set: ",loss," MAE: ",mae_mean,"+-",mae_std ," rMAE: ",rmae_mean,"+-",rmae_std)
                print("ACC",acc)                    
            # Flush
            sys.stdout.flush()
            sys.stderr.flush()
        # maybe adjust LR
        if eval_set == 'valInd' and not mdlParams['training_steps']:
            cond = (step-mdlParams['lastBestInd']) >= mdlParams['lowerLRAfter'] and (step-mdlParams['lastLRUpdate']) >= mdlParams['lowerLRAfter']
        else:
            cond = (mdlParams['lowerLRat'] + mdlParams['lowerLRAfter']*mdlParams['lastLRUpdate']) < step
        if cond:
            oldLR = modelVars['Sess'].run(lr)
            print("Old Learning Rate: ",oldLR)
            print("New Learning Rate: ",oldLR/mdlParams['LRstep'])
            update_op = lr.assign(oldLR/mdlParams['LRstep'])
            modelVars['Sess'].run(update_op)
            if eval_set == 'valInd' and not mdlParams['training_steps']:
                mdlParams['lastLRUpdate'] = step
            else:
                mdlParams['lastLRUpdate'] = mdlParams['lastLRUpdate']+1
    # After CV Training: print CV results and save them
    print("CV Set",cv+1)
    print("Best Loss:",allData['lossBest'][cv])
    print("Best MAE:",allData['maeBest'][cv],"+-",allData['maeStdBest'][cv])
    print("Best rMAE:",allData['rmaeBest'][cv],"+-",allData['rmaeStdBest'][cv])
    print("Best Acc:",allData['accBest'][cv])
    print("Convergence Steps:",allData['convergeTime'][cv])     
    # Write to File
    with open(mdlParams['saveDirBase'] + '/CV.pkl', 'wb') as f:
        pickle.dump(allData, f, pickle.HIGHEST_PROTOCOL)

# After training: evaluate
#cmd = os.getcwd()+"/eval_tfdata.py "+sys.argv[1]+" "+sys.argv[2]+" NONE "+sys.argv[3]
#os.system('{} {}'.format('python', cmd))
