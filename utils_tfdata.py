import functools
import numpy as np
import math
import tensorflow as tf
import h5py
from scipy import ndimage as nd
import time
import gc
import sklearn.preprocessing
from skimage.transform import resize
from skimage.transform import rotate
import itertools
from sklearn.metrics import auc, roc_curve, confusion_matrix, f1_score, jaccard_similarity_score
from tensorflow.python.ops import control_flow_ops
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Stuff for multi-GPU computing
def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ...  , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers.  So ..  we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def getErrForce_mgpu(mdlParams, indices, modelVars, placeholders, feed_list_inference):
    """Helper function to return the error of a set
    Args:
      mdlParams: dictionary, configuration file
      indices: string, either "trainInd", "valInd" or "testInd"
      modelVars: model tensors
      placeholders: potential placeholders
      feed_list_inference: corresponding list of values to feed for placeholders
    Returns:
      loss: float, avg loss
      mae: float, mean average error
      rmae: float, relative mean average error
      acc: float, average correlation coefficient
    """
    # Set up to-be-executed tensor list
    loss_list = []
    pred_list = []
    ind_list = []
    tar_list = []
    for i in range(len(mdlParams['numGPUs'])):
        loss_list.append(modelVars['loss_'+str(i)])
        pred_list.append(modelVars['pred_'+str(i)])
        ind_list.append(modelVars['Inds_'+str(i)])    
        tar_list.append(modelVars['Tar_'+str(i)])  
    loss = np.zeros([len(mdlParams[indices])])
    allInds = np.zeros([len(mdlParams[indices])],dtype=int)
    loss = np.zeros([len(mdlParams[indices])])
    for i in range(int(np.floor(len(mdlParams[indices])/len(mdlParams['numGPUs'])/mdlParams['batchSize']))):
        res_tuple = modelVars['Sess'].run([loss_list, pred_list, ind_list, tar_list], feed_dict={placeholders[p]: feed_list_inference[p] for p in placeholders})              
        # Write into proper arrays
        for k in range(len(mdlParams['numGPUs'])):    
            if i==0 and k==0:
                #print("pred shape",res_tuple[1][k].shape,"tar shape",res_tuple[3][k].shape)
                #loss = np.array(np.squeeze(res_tuple[0][k]))
                #print("loss",loss,loss.shape)
                predictions = res_tuple[1][k]
                targets = res_tuple[3][k]
                allInds[k+i*len(mdlParams['numGPUs'])] = res_tuple[2][k][0]
            else:
                #loss = np.concatenate((loss,np.squeeze(res_tuple[0][k])))
                predictions = np.concatenate((predictions,res_tuple[1][k]),0)
                targets = np.concatenate((targets,res_tuple[3][k]),0)
                allInds[k+i*len(mdlParams['numGPUs'])] = res_tuple[2][k][0]                           
    #print("Pred cropped",predictions_mc.shape)   
    print("NP Pred shape",predictions.shape,"Tar shape",targets.shape)                         
    # Average loss
    loss = np.mean(loss,0)
    # Rescale?
    if mdlParams['scale_targets']:
        targets = mdlParams['scaler'].inverse_transform(targets)
        predictions = mdlParams['scaler'].inverse_transform(predictions)
    # Get metrics
    # MAE
    mae_mean = np.mean(np.abs(predictions-targets),0)
    mae_std = np.std(np.abs(predictions-targets),0)    
    # Relative MAE
    tar_std = np.std(targets,0)
    rmae_mean = np.mean(np.abs(predictions-targets),0)/tar_std
    rmae_std = np.std(np.abs(predictions-targets),0)/tar_std
    # Avg. Corr. Coeff.
    corr = np.corrcoef(np.transpose(predictions),np.transpose(targets))
    # Extract relevant components for aCC
    acc = 0
    for k in range(mdlParams['numOut']):
        acc += corr[mdlParams['numOut']+k,k]
    acc /= mdlParams['numOut']             
    return loss, mae_mean, mae_std, rmae_mean, rmae_std, acc, predictions, targets, allInds

def preproc_train_force(input, flip_x, flip_y, rot_lat, is_4d):
    #input = tf.random_crop(input, [input_size[0],input_size[1],input_size[2],chan_size])
    # Flip
    if flip_x:
        rand_var_x = tf.random_uniform([], maxval = 1, dtype=tf.float32)
        input = tf.cond(rand_var_x > 0.5, lambda: input, lambda: tf.reverse(input,[1]))
    if flip_y:
        rand_var_y = tf.random_uniform([], maxval = 1, dtype=tf.float32)
        input = tf.cond(rand_var_y > 0.5, lambda: input, lambda: tf.reverse(input,[2]))
    # Rotate
    if rot_lat:
        if is_4d:
            rand_var_rot_lat = tf.random_uniform([], maxval = 1, dtype=tf.float32)
            input = tf.cond(rand_var_rot_lat > 0.75, lambda: input, 
                            lambda: tf.cond(rand_var_rot_lat > 0.5, lambda: tf.transpose(tf.reverse(input,[1]),[0,2,1,3,4]),
                            lambda: tf.cond(rand_var_rot_lat > 0.25, lambda: tf.reverse(input,[1,2]), lambda: tf.reverse(tf.transpose(input,[0,2,1,3,4]),[1]))))   
        else:
            rand_var_rot_lat = tf.random_uniform([], maxval = 1, dtype=tf.float32)
            input = tf.cond(rand_var_rot_lat > 0.75, lambda: input, 
                            lambda: tf.cond(rand_var_rot_lat > 0.5, lambda: tf.transpose(tf.reverse(input,[1]),[1,0,2,3]),
                            lambda: tf.cond(rand_var_rot_lat > 0.25, lambda: tf.reverse(input,[1,2]), lambda: tf.reverse(tf.transpose(input,[1,0,2,3]),[1]))))              
    return input

def preproc_val_force(input):
    # Right now, nothing  
    return input

def load_data(data_paths,labels_array, setids,local_indices, currInd, timesteps, h5datanames, target_shift, input_size):
    if len(input_size) == 3:
        if timesteps > 0:          
            # Label
            label = labels_array[currInd+target_shift,:]
            # Data
            with h5py.File(data_paths[setids[currInd]]) as f:
                data = np.transpose(f[h5datanames[0]].value)[(local_indices[currInd]-timesteps+1):(local_indices[currInd]+1),:,:,:]
                # Expand
                data = np.expand_dims(data,axis=4)
            for i in range(data.shape[0]):
                im_min = np.min(data[i,:,:,:,:])
                im_max = np.max(data[i,:,:,:,:])
                data[i,:,:,:,:] = (data[i,:,:,:,:] - im_min) / (im_max-im_min)                   
        else:
            # Label
            label = labels_array[currInd+target_shift,:]
            # Data
            with h5py.File(data_paths[setids[currInd]]) as f:
                data = np.transpose(f[h5datanames[0]].value)[local_indices[currInd],:,:,:]
                # Expand
                data = np.expand_dims(data,axis=3)
            im_min = np.min(data)
            im_max = np.max(data)
            data = (data - im_min) / (im_max-im_min)     
    elif len(input_size) == 2:        
        if timesteps > 0:          
            # Label
            label = labels_array[currInd+target_shift,:]
            # Data
            with h5py.File(data_paths[setids[currInd]]) as f:
                data = np.transpose(f[h5datanames[0]].value)[(local_indices[currInd]-timesteps+1):(local_indices[currInd]+1),:,:]
                # Expand
                data = np.expand_dims(data,axis=4)
            for i in range(data.shape[0]):
                im_min = np.min(data[i,:,:,:])
                im_max = np.max(data[i,:,:,:])
                data[i,:,:,:] = (data[i,:,:,:] - im_min) / (im_max-im_min)                
        else:
            # Label
            label = labels_array[currInd+target_shift,:]
            # Data
            with h5py.File(data_paths[setids[currInd]]) as f:
                data = f[h5datanames[0]].value[local_indices[currInd],:,:]
                # Expand
                data = np.expand_dims(data,axis=3)
            im_min = np.min(data)
            im_max = np.max(data)
            data = (data - im_min) / (im_max-im_min)       
    data = data.astype(np.float32)
    return data, label, currInd

def dataset_input_fn_force(mdlParams, mdlVars):
    """ Input function for 3d segmentation data from the decathalon
    Args:
        mdlParams: dict, config
        mdlVars: dict, tensorflow model variables
    Returns:
        iterator: an iterator that can be used to get dequeued batches
    """   
    with tf.device('/cpu:0'):
        # Lightweigth generator
        def lightweight_generator(indSet):
            if mdlParams['preload']:
                if indSet == 'trainInd':
                    set_range = itertools.cycle(range(len(mdlParams[indSet])))
                else:
                    set_range = range(len(mdlParams[indSet]))
                for i in set_range:
                    if i==0 and indSet == 'trainInd':
                        np.random.shuffle(mdlParams['trainInd'])
                    if len(mdlParams['data_array'].shape) == 5:
                        if 'timesteps' in mdlParams:
                            yield mdlParams['data_array'][(mdlParams[indSet][i]-mdlParams['timesteps']+1):(mdlParams[indSet][i]+1),:,:,:,:], mdlParams['labels_array'][mdlParams[indSet][i]+mdlParams['target_shift'],:], mdlParams[indSet][i]                                                                                                                  
                        else:
                            yield mdlParams['data_array'][mdlParams[indSet][i],:,:,:,:], mdlParams['labels_array'][mdlParams[indSet][i]+mdlParams['target_shift'],:], mdlParams[indSet][i]  
                    else:
                        if 'timesteps' in mdlParams:
                            #if indSet == 'testInd':
                            #    print("ind",mdlParams[indSet][i])
                            #    print("range",(mdlParams[indSet][i]-mdlParams['timesteps']+1),(mdlParams[indSet][i]+1))
                            yield mdlParams['data_array'][(mdlParams[indSet][i]-mdlParams['timesteps']+1):(mdlParams[indSet][i]+1),:,:,:], mdlParams['labels_array'][mdlParams[indSet][i]+mdlParams['target_shift'],:], mdlParams[indSet][i]
                        else:
                            yield mdlParams['data_array'][mdlParams[indSet][i],:,:,:], mdlParams['labels_array'][mdlParams[indSet][i]+mdlParams['target_shift'],:], mdlParams[indSet][i]                                                                                                                                   
            else:
                for i in range(len(mdlParams[indSet])):#itertools.cycle(range(len(mdlParams[indSet]))):
                    yield mdlParams[indSet][i]   
        # Actual data loading
        if not mdlParams['preload']:
            def data_loader(currInd):
                return tf.py_func(func = load_data, inp = (mdlParams['data_paths'], mdlParams['labels_array'], mdlParams['setids'],mdlParams['local_indices'], currInd, mdlParams['timesteps'] , mdlParams['h5datanames'], mdlParams['target_shift'], mdlParams['input_size']), Tout = (tf.float32,tf.float32,tf.int32))            
        def _set_shapes_out(x, y, ind):
            if 'timesteps' in mdlParams:
                if len(mdlParams['input_size']) == 3:
                    x.set_shape([mdlParams['timesteps'],mdlParams['input_size'][0],mdlParams['input_size'][1],mdlParams['input_size'][2],1])
                else:
                    x.set_shape([mdlParams['timesteps'],mdlParams['input_size'][0],mdlParams['input_size'][1],1])
            else:
                if len(mdlParams['input_size']) == 3:
                    x.set_shape([mdlParams['input_size'][0],mdlParams['input_size'][1],mdlParams['input_size'][2],1])
                else:
                    x.set_shape([mdlParams['input_size'][0],mdlParams['input_size'][1],1])
            #y.set_shape([mdlParams['numOut']])
            return x, y, ind      
        # Apply tensorflow level preprocessing
        def preproc_train_tf(x, y, ind):
            x = preproc_train_force(x, mdlParams['flip_x'], mdlParams['flip_y'], mdlParams['rot_lat'], 'timesteps' in mdlParams)
            return x, y, ind             
        def preproc_val_tf(x, y, ind):
            x = preproc_val_force(x)
            return x, y, ind            
        if mdlParams['preload']:    
            training_dataset = tf.data.Dataset.from_generator(lambda: lightweight_generator('trainInd'),output_types=(tf.float32,tf.float32,tf.int32))
        else:
            training_dataset = tf.data.Dataset.from_generator(lambda: lightweight_generator('trainInd'),output_types=(tf.int32))
        # Shuffle and repeat
        #training_dataset = training_dataset.shuffle(buffer_size=mdlParams['trainInd'].shape[0])
        # Load
        if not mdlParams['preload']:
            training_dataset = training_dataset.map(data_loader,num_parallel_calls=8)
        training_dataset = training_dataset.map(preproc_train_tf,num_parallel_calls=8)
        training_dataset = training_dataset.map(_set_shapes_out)            
        # batching
        training_dataset = training_dataset.batch(mdlParams['batchSize'],drop_remainder=True)
        #training_dataset = training_dataset.repeat()
        training_dataset = training_dataset.prefetch(1*mdlParams['batchSize'])        
        #*mdlParams['batchSize']
        # Same for validation/testing
        if 'valIndCV' in mdlParams or 'valInd' in mdlParams:
            # Define dataset
            if mdlParams['preload']: 
                validation_dataset = tf.data.Dataset.from_generator(lambda: lightweight_generator('valInd'),output_types=(tf.float32,tf.float32,tf.int32))
            else:
                validation_dataset = tf.data.Dataset.from_generator(lambda: lightweight_generator('valInd'),output_types=(tf.int32))
            validation_dataset = validation_dataset.repeat()                   
            validation_dataset = validation_dataset.prefetch(1*mdlParams['batchSize'])                
            # Load it
            if not mdlParams['preload']:
                validation_dataset = validation_dataset.map(data_loader,num_parallel_calls=8)
            # Preprocess
            validation_dataset = validation_dataset.map(preproc_val_tf,num_parallel_calls=8)  
            # Set output shapes
            validation_dataset = validation_dataset.map(_set_shapes_out)  
            validation_dataset = validation_dataset.batch(mdlParams['batchSize'],drop_remainder=True)  
            #if 'multiCropEval' not in mdlParams:
            #*mdlParams['batchSize']
        if mdlParams['print_trainerr']:
            # Define dataset
            if mdlParams['preload']: 
                train_eval_dataset = tf.data.Dataset.from_generator(lambda: lightweight_generator('trainInd_eval'),output_types=(tf.float32,tf.float32,tf.int32))
            else:
                train_eval_dataset = tf.data.Dataset.from_generator(lambda: lightweight_generator('trainInd_eval'),output_types=(tf.int32))
            # Load it
            if not mdlParams['preload']:
                train_eval_dataset = train_eval_dataset.map(data_loader,num_parallel_calls=8)
            # Preprocess
            train_eval_dataset = train_eval_dataset.map(preproc_val_tf,num_parallel_calls=8)  
            # Set output shapes
            train_eval_dataset = train_eval_dataset.map(_set_shapes_out)  
            train_eval_dataset = train_eval_dataset.batch(mdlParams['batchSize'],drop_remainder=True)  
            #if 'multiCropEval' not in mdlParams:
            train_eval_dataset = train_eval_dataset.repeat()                   
            train_eval_dataset = train_eval_dataset.prefetch(1)              
        if 'testInd' in mdlParams:
            # Define dataset
            if mdlParams['preload']: 
                test_dataset = tf.data.Dataset.from_generator(lambda: lightweight_generator('testInd'),output_types=(tf.float32,tf.float32,tf.int32))
            else:
                test_dataset = tf.data.Dataset.from_generator(lambda: lightweight_generator('testInd'),output_types=(tf.int32))
            # Load it
            if not mdlParams['preload']:
                test_dataset = test_dataset.map(data_loader,num_parallel_calls=8)
            # Preprocess
            test_dataset = test_dataset.map(preproc_val_tf,num_parallel_calls=8)  
            # Set output shapes
            test_dataset = test_dataset.map(_set_shapes_out)  
            test_dataset = test_dataset.batch(mdlParams['batchSize'],drop_remainder=True)  
            #if 'multiCropEval' not in mdlParams:
            test_dataset = test_dataset.repeat()                   
            test_dataset = test_dataset.prefetch(1) #*mdlParams['batchSize']

        mdlVars['handle'] = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(mdlVars['handle'], training_dataset.output_types, training_dataset.output_shapes)
        # Dequeue
        #features, labels = iterator.get_next()
        # Iterator for each dataset
        training_iterator = training_dataset.make_one_shot_iterator()
        if 'valIndCV' in mdlParams or 'valInd' in mdlParams:
            validation_iterator = validation_dataset.make_one_shot_iterator()
        if mdlParams['print_trainerr']:
            train_eval_iterator = train_eval_dataset.make_one_shot_iterator()
        if 'testInd' in mdlParams:
            test_iterator = test_dataset.make_one_shot_iterator()
        # Handels to switch inbetween these iterators
        mdlVars['training_handle'] = mdlVars['Sess'].run(training_iterator.string_handle())
        #mdlVars['training_eval_handle'] = mdlVars['Sess'].run(training_eval_iterator.string_handle())
        if 'valIndCV' in mdlParams or 'valInd' in mdlParams:
            mdlVars['validation_handle'] = mdlVars['Sess'].run(validation_iterator.string_handle())
        if mdlParams['print_trainerr']:
            mdlVars['train_eval_handle'] = mdlVars['Sess'].run(train_eval_iterator.string_handle())
        if 'testInd' in mdlParams:
            mdlVars['test_handle'] = mdlVars['Sess'].run(test_iterator.string_handle())    
        return iterator  

def getInputFunction(mdlParams,modelVars):
  """Returns a function for a getPaths function
  Args:
    mdlParams: dictionary, contains configuration
  Returns:
    getPaths: A function that returns a the paths for images and labels
  Raises:
    ValueError: If network name is not recognized.
  """
  if mdlParams['input_func_type'] not in inputFunc_map:
    raise ValueError('Name of getPaths function unknown %s' % mdlParams['input_func_type'])
  func = inputFunc_map[mdlParams['input_func_type']]
  @functools.wraps(func)
  def getInputFcn():
      return func(mdlParams,modelVars)
  return getInputFcn

inputFunc_map = {'force': dataset_input_fn_force,
               }       
            