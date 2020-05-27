import tensorflow as tf
from tensorflow.python.ops import math_ops
import tensorflow.contrib.slim as slim
from tensorflow.contrib import rnn
import numbers
import numpy as np
import functools
import h5py
import math
import conv4d

def resneta_block(x, numFmOut, stride, factorize=False, kernel_size=3,is_temporal=False):
    """Defines a single resnetA block, according to paper
    Args: 
      x: block input, 5D tensor
      base_fm: base number of feature maps in the block
    Returns:
      output: 5D tensor, output of the block 
    """
    # Number of input fms
    numFmIn = x.get_shape().as_list()[-1]
    # Determine if its a reduction
    if numFmOut > numFmIn:
        increase_dim = True
    else:
        increase_dim = False
    # First 3x3 layer
    with tf.variable_scope('conv3x3x3_1'):
        if factorize:
            layer = slim.convolution(x,numFmOut,[1,kernel_size,kernel_size],stride=stride)
            layer = slim.convolution(layer,numFmOut,[kernel_size,1,1],stride=1)
        else:
            layer = slim.convolution(x,numFmOut,kernel_size,stride=stride)
    # Second 3x3 layer, no activation, only bnorm
    with tf.variable_scope('conv3x3x3_2'):
        if factorize:
            layer = slim.convolution(layer,numFmOut,[1,kernel_size,kernel_size],stride=1, activation_fn=tf.nn.relu)
            layer = slim.convolution(layer,numFmOut,[kernel_size,1,1],stride=1, activation_fn=None)
        else:
            layer = slim.convolution(layer,numFmOut,kernel_size,stride=1, activation_fn=None)
    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    # depth of input layers
    adjusted_input = x
    if stride[1] == 2:
        # take care of 1D<->2D<->3D
        if len(x.get_shape().as_list()) == 5:
            if is_temporal:
                adjusted_input = tf.nn.pool(adjusted_input,[2,2,2], "AVG", padding='SAME', strides=[1,2,2])
            else:
                adjusted_input = tf.nn.pool(adjusted_input,[2,2,2], "AVG", padding='SAME', strides=[2,2,2])
        elif len(x.get_shape().as_list()) == 4:
            adjusted_input = tf.nn.pool(adjusted_input,[2,2], "AVG", padding='SAME', strides=[2,2])
        else:
            adjusted_input = tf.nn.pool(adjusted_input,[2], "AVG", padding='SAME', strides=[2])
    if increase_dim:
        lower_pad = math.ceil((numFmOut-numFmIn)/2)
        upper_pad = (numFmOut-numFmIn)-lower_pad
        # take care of 1D<->2D<->3D
        if len(x.get_shape().as_list()) == 5:
            adjusted_input = tf.pad(adjusted_input, [[0, 0], [0, 0], [0, 0], [0,0], [lower_pad,upper_pad]])
        elif len(x.get_shape().as_list()) == 4:
            adjusted_input = tf.pad(adjusted_input, [[0, 0], [0, 0], [0, 0], [lower_pad,upper_pad]])
        else:
            adjusted_input = tf.pad(adjusted_input, [[0, 0], [0, 0], [lower_pad,upper_pad]])
    # Residual connection + activation
    output = tf.nn.relu(adjusted_input + layer)
    return output

def resneta_block_4d(x, numFmOut, stride, batchSize, placeholders, factorize, kernel_size=[3,3,3,3]):
    """Defines a single resnetA block, according to paper
    Args: 
      x: block input, 5D tensor
      base_fm: base number of feature maps in the block
    Returns:
      output: 5D tensor, output of the block 
    """
    # Number of input fms
    numFmIn = x.get_shape().as_list()[1]
    #print(x.shape)
    # Determine if its a reduction
    if numFmOut > numFmIn:
        increase_dim = True
    else:
        increase_dim = False
    # First 3x3 layer
    with tf.variable_scope('conv3x3x3x3_1'):
        if factorize:
            layer = conv4d.conv4d_BatchNorm(input=x, filters=numFmOut, kernel_size=[1,kernel_size[1],kernel_size[2],kernel_size[3]], strides=stride, activation=tf.nn.relu, data_format='channels_first', name='conv1_spatial', BATCH = batchSize, placeholders=placeholders)
            layer = conv4d.conv4d_BatchNorm(input=layer, filters=numFmOut, kernel_size=[kernel_size[0],1,1,1], strides=[1,1,1,1], activation=tf.nn.relu, data_format='channels_first', name='conv1_temp', BATCH = batchSize, placeholders=placeholders)    
        else:
            layer = conv4d.conv4d_BatchNorm(input=x, filters=numFmOut, kernel_size=kernel_size, strides=stride, activation=tf.nn.relu, data_format='channels_first', name='conv1', BATCH = batchSize, placeholders=placeholders)
    # Second 3x3 layer, no activation, only bnorm
    with tf.variable_scope('conv3x3x3x3_2'):
        if factorize:
            layer = conv4d.conv4d_BatchNorm(input=layer, filters=numFmOut, kernel_size=[1,kernel_size[1],kernel_size[2],kernel_size[3]], strides=[1,1,1,1], activation=tf.nn.relu, data_format='channels_first', name='conv1_spatial', BATCH = batchSize, placeholders=placeholders)        
            layer = conv4d.conv4d_BatchNorm(input=layer, filters=numFmOut, kernel_size=[kernel_size[0],1,1,1], strides=[1,1,1,1], activation=None, data_format='channels_first', name='conv1_temp', BATCH = batchSize, placeholders=placeholders)                
        else:
            layer = conv4d.conv4d_BatchNorm(input=layer, filters=numFmOut, kernel_size=kernel_size, strides=[1,1,1,1], activation=None, data_format='channels_first', name='conv1', BATCH = batchSize, placeholders=placeholders)        
    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    # depth of input layers
    adjusted_input = x
    if stride[1] == 2:
        # take care of 4D
        adjusted_input = conv4d.pool4d(input=adjusted_input, kernel_size=[2, 2, 2, 2], strides=[1,2,2,2], name ='adjusted_input_1', BATCH=batchSize)
    if increase_dim:
        lower_pad = math.ceil((numFmOut-numFmIn)/2)
        upper_pad = (numFmOut-numFmIn)-lower_pad
        # take care of 4D
        adjusted_input = tf.pad(adjusted_input, [[0, 0],[lower_pad,upper_pad], [0, 0], [0, 0], [0,0],[0,0],])
    # Residual connection + activation
    output = tf.nn.relu(adjusted_input + layer)
    return output    

# Thanks to https://github.com/OlavHN/bnlstm for part of the code
def batch_norm(inputs, name_scope, is_training, epsilon=1e-3, decay=0.99, keep_time_dim=False):
    """Batch normalization for recurrent architectures
    Args:
      inputs: input tensor
      name_scope: scope for all variables and operations
      is_training: indicates if it is training or evaluation
      epsilon: batch norm param, numerical stability
      decay: batch norm param, decay rate of moving averages
      keep_time_dim: defines, whether mean/var paramters are tracked for all timesteps independently or not 
    Returns:
      output: tensor of same size as input tensor
    """
    with tf.variable_scope(name_scope):
        if keep_time_dim:
            size = [1]
            size.extend(inputs.get_shape().as_list()[1:])
            size_lin = [inputs.get_shape().as_list()[-1]]
        else:
            size = [inputs.get_shape().as_list()[-1]]
            size_lin = size
        with tf.device('/cpu:0'):
            scale = tf.get_variable(
                'scale', size_lin, initializer=tf.constant_initializer(0.1))
            offset = tf.get_variable('offset', size_lin)

            population_mean = tf.get_variable(
                'population_mean', size,
                initializer=tf.zeros_initializer(), trainable=False)
            population_var = tf.get_variable(
                'population_var', size,
                initializer=tf.ones_initializer(), trainable=False)
        if keep_time_dim:
            batch_mean, batch_var = tf.nn.moments(inputs, [0])
            batch_mean = tf.expand_dims(batch_mean,0)
            batch_var = tf.expand_dims(batch_var,0)
        else:
            batch_mean, batch_var = tf.nn.moments(inputs, [0,1])
        # The following part is based on the implementation of :
        # https://github.com/cooijmanstim/recurrent-batch-normalization
        train_mean_op = tf.assign(
            population_mean,
            population_mean * decay + batch_mean * (1 - decay))
        train_var_op = tf.assign(
            population_var, population_var * decay + batch_var * (1 - decay))

        if is_training is True:
            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.nn.batch_normalization(
                    inputs, batch_mean, batch_var, offset, scale, epsilon)
        else:
            return tf.nn.batch_normalization(
                inputs, population_mean, population_var, offset, scale,
                epsilon)

class BN_GRUCell(tf.nn.rnn_cell.RNNCell):
  """GRU cell with Recurrent Batch Normalization. """
  def __init__(self, num_hidden, activation=tf.tanh, reuse=None, is_training = None, keep_time_dim = False):
    """ Creates a batch normalized gated current unit cell
    Args:
      shape: number of features in the input data
      filters: number of feature maps to be computed
      kernel: kernel size
      activation: activation of non-gates
      reuse: reuse for paramter sharing
      is_training: determines training state for batch norm
      keep_time_dim: determines whether mean/var in batch norm are tracked per time step or not
    """
    super(BN_GRUCell, self).__init__(_reuse=reuse)
    self._num_hidden = num_hidden
    self._size = tf.TensorShape([self._num_hidden])
    self._activation = activation
    self._is_training = is_training
    self._keep_time_dim = keep_time_dim

  @property
  def state_size(self):
    return self._size

  @property
  def output_size(self):
    return self._size

  def __call__(self, x, h, scope=None):
      with slim.arg_scope([slim.fully_connected],activation_fn = None, weights_initializer=tf.truncated_normal_initializer(stddev=0.01)):
          with tf.variable_scope('gates'):
              # Compute two convolutions in one operation
              m = 2 * self._num_hidden
              # Convolve input and state
              y_x = slim.fully_connected(x, m, scope='mul_x', biases_initializer=None)
              y_h = slim.fully_connected(h, m, scope='muk_h')
              # Normalize input only
              bn_x = batch_norm(y_x, 'bn_x', self._is_training, keep_time_dim=self._keep_time_dim)
              bn_h = y_h
              # Combine
              gru_matrix = bn_x + bn_h
              # Split for different gates
              r, u = tf.split(value=gru_matrix, num_or_size_splits=2, axis=1)       
              r, u = tf.sigmoid(r), tf.sigmoid(u)
          with tf.variable_scope('candidate'):
              # These cannot be done jointly as BN needs to be applied to one of them
              m = self._num_hidden
              y_x_c = slim.fully_connected(x, m, scope='muk_x_c', biases_initializer=None)
              y_rh = slim.fully_connected(r*h, m, scope='muk_rh')
              # Apply batch norm to input, again
              bn_x_c = batch_norm(y_x_c, 'bn_x_c', self._is_training, keep_time_dim = self._keep_time_dim)
              # Combine
              y = bn_x_c + y_rh
              # Final aggregation
              h = u * h + (1 - u) * self._activation(y)
      # Keep compatibility with lstm format
      return h, h

class BN_LSTMCell(tf.nn.rnn_cell.RNNCell):
  """LSTM cell with Recurrent Batch Normalization. """
  def __init__(self, num_hidden, forget_bias = 1.0, activation=tf.tanh, reuse=None, is_training = None, keep_time_dim = False):
    """ Creates a batch normalized gated current unit cell
    Args:
      shape: number of features in the input data
      filters: number of feature maps to be computed
      kernel: kernel size
      activation: activation of non-gates
      reuse: reuse for paramter sharing
      is_training: determines training state for batch norm
      keep_time_dim: determines whether mean/var in batch norm are tracked per time step or not
    """
    super(BN_LSTMCell, self).__init__(_reuse=reuse)
    self._num_hidden = num_hidden
    self._size = tf.TensorShape([self._num_hidden])
    self._forget_bias = forget_bias
    self._activation = activation
    self._is_training = is_training
    self._keep_time_dim = keep_time_dim

  @property
  def state_size(self):
    return tf.nn.rnn_cell.LSTMStateTuple(self._size, self._size)

  @property
  def output_size(self):
    return self._size

  def __call__(self, x, state, scope=None):
      with slim.arg_scope([slim.fully_connected],activation_fn = None, weights_initializer=tf.truncated_normal_initializer(stddev=0.01)):
          with tf.variable_scope('gates'):
              # Get state        
              c, h = state         
              print("x shape",x.shape)
              print("h shape",h.shape)                         
              # Compute two convolutions in one operation
              m = 4 * self._num_hidden
              # Convolve input and state
              y_x = slim.fully_connected(x, m, scope='mul_x', biases_initializer=None)
              y_h = slim.fully_connected(h, m, scope='muk_h')
              # Normalize input only
              bn_x = batch_norm(y_x, 'bn_x', self._is_training, keep_time_dim=self._keep_time_dim)
              bn_h = y_h
              # Combine
              lstm_matrix = bn_x + bn_h
              # Split for different gates
              j, i, f, o = tf.split(value=lstm_matrix, num_or_size_splits=4, axis=1)    
              # Build gates
              f = tf.sigmoid(f + self._forget_bias)
              i = tf.sigmoid(i)
              c = c*f + i*self._activation(j)
              # Output
              o = tf.sigmoid(o)
              h = o*self._activation(c)
              # Build state
              state = tf.nn.rnn_cell.LSTMStateTuple(c,h)              
      return h, state

class BN_ConvGRUCell(tf.nn.rnn_cell.RNNCell):
  """A GRU cell with convolutions instead of multiplications."""

  def __init__(self, shape, filters, kernel, activation=tf.tanh, reuse=None, is_training = None, keep_time_dim = False):
    """ Creates a batch normalized convolutional gated current unit cell
    Args:
      shape: number of features in the input data
      filters: number of feature maps to be computed
      kernel: kernel size
      activation: activation of non-gates
      reuse: reuse for paramter sharing
      is_training: determines training state for batch norm
      keep_time_dim: determines whether mean/var in batch norm are tracked per time step or not
    """
    super(BN_ConvGRUCell, self).__init__(_reuse=reuse)
    self._filters = filters
    self._kernel = kernel
    if len(shape) == 3:
        self._size = tf.TensorShape([shape[0], shape[1], shape[2], self._filters])
    elif len(shape) == 2:
        self._size = tf.TensorShape([shape[0], shape[1], self._filters])
    self._activation = activation
    self._is_training = is_training
    self._keep_time_dim = keep_time_dim

  @property
  def state_size(self):
    return self._size

  @property
  def output_size(self):
    return self._size

  def __call__(self, x, h, scope=None):
    with slim.arg_scope([slim.convolution], weights_initializer=tf.truncated_normal_initializer(stddev=0.01), activation_fn = None):
        with tf.variable_scope('gates'):
          # Compute two convolutions in one operation
          m = 2 * self._filters
          # Convolve input and state
          print("x shape",x.shape)
          print("h shape",h.shape)
          y_x = slim.convolution(x, m, self._kernel, scope='conv_x', biases_initializer=None)
          y_h = slim.convolution(h, m, self._kernel, scope='conv_h')
          # Normalize input only
          bn_x = batch_norm(y_x, 'bn_x', self._is_training, keep_time_dim=self._keep_time_dim)
          bn_h = y_h
          # Combine
          gru_matrix = bn_x + bn_h
          # Split for different gates
          r, u = tf.split(value=gru_matrix, num_or_size_splits=2, axis=len(x.shape)-1)       
          r, u = tf.sigmoid(r), tf.sigmoid(u)
        with tf.variable_scope('candidate'):
          # These cannot be done jointly as BN needs to be applied to one of them
          m = self._filters
          y_x_c = slim.convolution(x, m, self._kernel, scope='conv_x_c', biases_initializer=None)
          y_rh = slim.convolution(r*h, m , self._kernel, scope='conv_rh')
          # Apply batch norm to input, again
          bn_x_c = batch_norm(y_x_c, 'bn_x_c', self._is_training, keep_time_dim = self._keep_time_dim)
          # Combine
          y = bn_x_c + y_rh
          # Final aggregation
          h = u * h + (1 - u) * self._activation(y)
        # Keep compatibility with lstm format
    return h, h

class BN_ConvLSTMCell(tf.nn.rnn_cell.RNNCell):
  """A GRU cell with convolutions instead of multiplications."""

  def __init__(self, shape, filters, kernel, forget_bias=1.0, activation=tf.tanh, reuse=None, is_training = None, keep_time_dim = False):
    """ Creates a batch normalized convolutional gated current unit cell
    Args:
      shape: number of features in the input data
      filters: number of feature maps to be computed
      kernel: kernel size
      activation: activation of non-gates
      reuse: reuse for paramter sharing
      is_training: determines training state for batch norm
      keep_time_dim: determines whether mean/var in batch norm are tracked per time step or not
    """
    super(BN_ConvLSTMCell, self).__init__(_reuse=reuse)
    self._filters = filters
    self._kernel = kernel
    self._forget_bias = forget_bias
    if len(shape) == 3:
        self._size = tf.TensorShape([shape[0], shape[1], shape[2], self._filters])
    elif len(shape) == 2:
        self._size = tf.TensorShape([shape[0], shape[1], self._filters])
    self._activation = activation
    self._is_training = is_training
    self._keep_time_dim = keep_time_dim

  @property
  def state_size(self):
    return tf.nn.rnn_cell.LSTMStateTuple(self._size, self._size)

  @property
  def output_size(self):
    return self._size

  def __call__(self, x, state, scope=None):
    with slim.arg_scope([slim.convolution], weights_initializer=tf.truncated_normal_initializer(stddev=0.01), activation_fn = None):
        with tf.variable_scope('gates'):
          # Get state
          c, h = state
          # Compute two convolutions in one operation
          m = 4 * self._filters
          # Convolve input and state
          print("x shape",x.shape)
          print("h shape",h.shape)
          y_x = slim.convolution(x, m, self._kernel, scope='conv_x', biases_initializer=None)
          y_h = slim.convolution(h, m, self._kernel, scope='conv_h')
          # Normalize input only
          bn_x = batch_norm(y_x, 'bn_x', self._is_training, keep_time_dim=self._keep_time_dim)
          bn_h = y_h
          # Combine
          lstm_matrix = bn_x + bn_h
          # Split for different gates
          j, i, f, o = tf.split(value=lstm_matrix, num_or_size_splits=4, axis=len(x.shape)-1)    
          # Build gates
          f = tf.sigmoid(f + self._forget_bias)
          i = tf.sigmoid(i)
          c = c*f + i*self._activation(j)
          # Output
          o = tf.sigmoid(o)
          h = o*self._activation(c)
          # Build state
          state = tf.nn.rnn_cell.LSTMStateTuple(c,h)
    return h, state

def Resnet4D(x, mdlParams, placeholders=None):
    """ Defines the 4D CNN Architecture, based on Resnet 
    Args:
      x: 6D input tensor, usually a placeholder of shape [batchSize, timesteps, [width, height, depth], channel]
      mdlParams: dictionary, contains model configuration
      is_training: boolean, indicates if it is training or evaluation
    Returns:
      output: 2D tensor of shape [batchSize, numberOfOutputs]
    """
    with tf.variable_scope('ResNetA4D'):
        with tf.variable_scope('Initial'):
            # Transform to channel first
            layer = tf.transpose(x,[0,5,1,2,3,4])        
            if mdlParams['factorize']:
                layer = conv4d.conv4d_BatchNorm(input=layer, filters=mdlParams['num_filters_init'], kernel_size=[1,mdlParams['kernel_init'][1],mdlParams['kernel_init'][2],mdlParams['kernel_init'][3]], strides=mdlParams['strides_init'], data_format='channels_first', name='conv1_spatial', BATCH = mdlParams['batchSize'], placeholders=placeholders)
                layer = conv4d.conv4d_BatchNorm(input=layer, filters=mdlParams['num_filters_init'], kernel_size=[mdlParams['kernel_init'][0],1,1,1], strides=[1,1,1,1], data_format='channels_first', name='conv1_temp', BATCH = mdlParams['batchSize'], placeholders=placeholders)           
            else:                        
                layer = conv4d.conv4d_BatchNorm(input=layer, filters= mdlParams['num_filters_init'], kernel_size=mdlParams['kernel_init'], strides=mdlParams['strides_init'], data_format='channels_first', name='conv1', BATCH = mdlParams['batchSize'], placeholders=placeholders)
            print(layer.name,layer.get_shape())
            # Resnet modules
        with tf.variable_scope('Resnet_modules'):
            # Iterate through all modules
            for i in range(len(mdlParams['ResNet_Size'])):
                with tf.variable_scope('Module_%d'%(i)):
                    # Iterate through all blocks inside the module
                    for j in range(mdlParams['ResNet_Size'][i]):
                        with tf.variable_scope('Block_%d'%(j)):
                            # Set desired output feature map dimension of the block and the desired stride for the first block in the module
                            if j==0:
                                output_fm = mdlParams['ResNet_FM'][i]
                                block_stride = mdlParams['ResNet_Stride'][i]
                            else:
                                block_stride = [1,1,1,1]
                            layer = resneta_block_4d(layer, output_fm, block_stride, mdlParams['batchSize'],placeholders,mdlParams['factorize'])
                            print(layer.name,layer.get_shape())
            # GAP with channel first
            if tf.__version__ == '1.14.0':
                layer = math_ops.reduce_mean(layer, axis=[2,3,4,5], keepdims = False, name='global_pool')
            else:
                layer = math_ops.reduce_mean(layer, axis=[2,3,4,5], keep_dims = False, name='global_pool')
            print(layer.name,layer.get_shape())
            # FC-Layer
            output = slim.layers.fully_connected(layer, mdlParams['numOut'] , activation_fn=None,weights_initializer =tf.truncated_normal_initializer(stddev=0.01))
    return output    

def Resnet3D(x, mdlParams, placeholders=None):
    """ Defines the 3D CNN Architecture, based on Resnet 
    Args:
      x: 3D input tensor, usually a placeholder of shape [batchSize, timesteps, [width, height], channel]
      mdlParams: dictionary, contains model configuration
      is_training: boolean, indicates if it is training or evaluation
    Returns:
      output: 2D tensor of shape [batchSize, numberOfOutputs]
    """
    with tf.variable_scope('ResNetA3D'):
        with slim.arg_scope([slim.convolution], padding='SAME', activation_fn=tf.nn.relu, weights_initializer=tf.truncated_normal_initializer(stddev=0.01), normalizer_fn=slim.batch_norm, normalizer_params={'is_training':placeholders['train_state'], 'epsilon':0.0001, 'decay':0.9,  'center':True, 'scale':True, 'activation_fn':None, 'updates_collections':tf.GraphKeys.UPDATE_OPS, 'fused': False}):
            #with slim.arg_scope([slim.batch_norm],is_training=placeholders['train_state'], epsilon=0.0001, decay=0.9,  center=True, scale=True, activation_fn=None, updates_collections=tf.GraphKeys.UPDATE_OPS, fused = False):
                # Initial part
                with tf.variable_scope('Initial'):
                    layer = slim.convolution(x, mdlParams['num_filters_init'], mdlParams['kernel_init'], stride=mdlParams['strides_init'], scope='conv1')
                    print(layer.name,layer.get_shape())
                # Resnet modules
                with tf.variable_scope('Resnet_modules'):
                    # Iterate through all modules
                    for i in range(len(mdlParams['ResNet_Size'])):
                        with tf.variable_scope('Module_%d'%(i)):
                            # Iterate through all blocks inside the module
                            for j in range(mdlParams['ResNet_Size'][i]):
                                with tf.variable_scope('Block_%d'%(j)):
                                    # Set desired output feature map dimension of the block and the desired stride for the first block in the module
                                    if j==0:
                                        output_fm = mdlParams['ResNet_FM'][i]
                                        block_stride = mdlParams['ResNet_Stride'][i]
                                    else:
                                        if len(layer.get_shape().as_list()) == 5:
                                            block_stride = [1,1,1]
                                        else:
                                            block_stride = [1,1]
                                    layer = resneta_block(layer, output_fm, block_stride, factorize= mdlParams['factorize'],is_temporal = 'timesteps' in mdlParams)
                                    print(layer.name,layer.get_shape())
                # GAP
                if len(layer.get_shape().as_list()) == 3:
                    if tf.__version__ == '1.14.0':
                        layer = math_ops.reduce_mean(layer, axis=[1], keepdims = False, name='global_pool')
                    else:
                        layer = math_ops.reduce_mean(layer, axis=[1], keep_dims = False, name='global_pool')                            
                elif len(layer.get_shape().as_list()) == 4:
                    if tf.__version__ == '1.14.0':
                        layer = math_ops.reduce_mean(layer, axis=[1,2], keepdims = False, name='global_pool')
                    else:
                        layer = math_ops.reduce_mean(layer, axis=[1,2], keep_dims = False, name='global_pool')                     
                elif len(layer.get_shape().as_list()) == 5:
                    if tf.__version__ == '1.14.0':
                        layer = math_ops.reduce_mean(layer, axis=[1,2,3], keepdims = False, name='global_pool')
                    else:
                        layer = math_ops.reduce_mean(layer, axis=[1,2,3], keep_dims = False, name='global_pool')                     
                print(layer.name,layer.get_shape())
                # FC-Layer
                output = slim.layers.fully_connected(layer, mdlParams['numOut'] , activation_fn=None,weights_initializer =tf.truncated_normal_initializer(stddev=0.01))
    return output        

def convGRUCNN(x, mdlParams, placeholders=None):
    """Defines the convGRU-CNN architecture from the paper "Needle Tip Force Estimation Using a Single OCT Fiber and a convGRU-CNN Architecture"
    Args:
      x: 4D-6D input tensor, usually a placeholder of shape [batchSize, timesteps, [width, height, depth], channel]
      mdlParams: dictionary, contains model configuration
      is_training: boolean, indicates if it is training or evaluation
    Returns:
      output: 2D tensor of shape [batchSize, numberOfOutputs]
    """
    # First: convGRU blocks
    all_cells = [None] * mdlParams['num_convGRU_layers']
    for i in range(mdlParams['num_convGRU_layers']):
        if mdlParams.get('use_lstm',False):
            all_cells[i] = BN_ConvLSTMCell(mdlParams['input_size'],mdlParams['num_hidden'],mdlParams['convgru_kernel'],keep_time_dim=mdlParams['keep_time_dim'],is_training=placeholders['train_state'])
        else:
            all_cells[i] = BN_ConvGRUCell(mdlParams['input_size'],mdlParams['num_hidden'],mdlParams['convgru_kernel'],keep_time_dim=mdlParams['keep_time_dim'],is_training=placeholders['train_state'])
        #single_cell = ConvGRU.BN_ConvGRUCell([mdlParams['num_features']],mdlParams['num_hidden'],mdlParams['convgru_kernel'],keep_time_dim=mdlParams['keep_time_dim'])
        # Dropout wrapper
        all_cells[i] = rnn.DropoutWrapper(all_cells[i],input_keep_prob=placeholders['KP1'],output_keep_prob=placeholders['KP2'])
        # Stack layers
    convgru_cells = rnn.MultiRNNCell(all_cells)
    # Get last output over time from GRU cels
    output_gru, _ = tf.nn.dynamic_rnn(convgru_cells, x, dtype=tf.float32)
    # Different sizes
    if len(output_gru.get_shape().as_list()) == 4:
        output_gru = tf.transpose(output_gru, [1, 0, 2, 3])
    elif len(output_gru.get_shape().as_list()) == 5:
        output_gru = tf.transpose(output_gru, [1, 0, 2, 3, 4])
    elif len(output_gru.get_shape().as_list()) == 6:
        output_gru = tf.transpose(output_gru, [1, 0, 2, 3, 4, 5])        
    print("out shape",output_gru.get_shape())
    last = tf.gather(output_gru, int(output_gru.get_shape()[0]) - 1)
    print("Last shape",last.get_shape())
    # Feed into CNN model, Resnet based
    with tf.variable_scope('CNN'):
        with slim.arg_scope([slim.convolution], padding='SAME', activation_fn=tf.nn.relu, weights_initializer=tf.truncated_normal_initializer(stddev=0.01), normalizer_fn=slim.batch_norm, normalizer_params={'is_training':placeholders['train_state'], 'epsilon':0.0001, 'decay':0.9,  'center':True, 'scale':True, 'activation_fn':None, 'updates_collections':tf.GraphKeys.UPDATE_OPS, 'fused': False}):
            #with slim.arg_scope([slim.batch_norm],is_training=placeholders['train_state'], epsilon=0.0001, decay=0.9,  center=True, scale=True, activation_fn=None, updates_collections=tf.GraphKeys.UPDATE_OPS, fused = False):
                # Initial part
                with tf.variable_scope('Initial'):
                    layer = slim.convolution(last, mdlParams['num_filters_init'], mdlParams['kernel_init'], stride=mdlParams['strides_init'], scope='conv1')
                # Resnet modules
                with tf.variable_scope('Resnet_modules'):
                    # Iterate through all modules
                    for i in range(len(mdlParams['ResNet_Size'])):
                        with tf.variable_scope('Module_%d'%(i)):
                            # Iterate through all blocks inside the module
                            for j in range(mdlParams['ResNet_Size'][i]):
                                with tf.variable_scope('Block_%d'%(j)):
                                    # Set desired output feature map dimension of the block and the desired stride for the first block in the module
                                    if j==0:
                                        output_fm = mdlParams['ResNet_FM'][i]
                                        block_stride = mdlParams['ResNet_Stride'][i]
                                    else:
                                        if len(mdlParams['input_size']) == 3:
                                            block_stride = [1,1,1]
                                        elif len(mdlParams['input_size']) == 2:
                                            block_stride = [1,1]
                                    if len(mdlParams['input_size']) == 3:
                                        layer = resneta_block(layer, output_fm, block_stride, kernel_size=[3,3,3])
                                    else:
                                        layer = resneta_block(layer, output_fm, block_stride, kernel_size=[3,3])
                # GAP
                if len(layer.get_shape().as_list()) == 3:
                    if tf.__version__ == '1.14.0':
                        layer = math_ops.reduce_mean(layer, axis=[1], keepdims = False, name='global_pool')
                    else:
                        layer = math_ops.reduce_mean(layer, axis=[1], keep_dims = False, name='global_pool')                            
                elif len(layer.get_shape().as_list()) == 4:
                    if tf.__version__ == '1.14.0':
                        layer = math_ops.reduce_mean(layer, axis=[1,2], keepdims = False, name='global_pool')
                    else:
                        layer = math_ops.reduce_mean(layer, axis=[1,2], keep_dims = False, name='global_pool')                     
                elif len(layer.get_shape().as_list()) == 5:
                    if tf.__version__ == '1.14.0':
                        layer = math_ops.reduce_mean(layer, axis=[1,2,3], keepdims = False, name='global_pool')
                    else:
                        layer = math_ops.reduce_mean(layer, axis=[1,2,3], keep_dims = False, name='global_pool')                     
                # FC-Layer
                output = slim.layers.fully_connected(layer, mdlParams['numOut'] , activation_fn=None,weights_initializer =tf.truncated_normal_initializer(stddev=0.01))
    return output

def CNNGRU(x, mdlParams, placeholders=None):
    """Defines the CNN-GRU architecture from the paper "Needle Tip Force Estimation Using a Single OCT Fiber and a convGRU-CNN Architecture"
    Args:
      x: 4D-6D input tensor, usually a placeholder of shape [batchSize, timesteps, [width, height, depth], channel]
      mdlParams: dictionary, contains model configuration
      is_training: boolean, indicates if it is training or evaluation
    Returns:
      output: 2D tensor of shape [batchSize, numberOfOutputs]
    """
    # First: CNN feature extraction
    with tf.variable_scope('CNN'):
        with slim.arg_scope([slim.convolution], padding='SAME', activation_fn=tf.nn.relu, weights_initializer=tf.truncated_normal_initializer(stddev=0.01), normalizer_fn=slim.batch_norm, normalizer_params={'is_training':placeholders['train_state'], 'epsilon':0.0001, 'decay':0.9,  'center':True, 'scale':True, 'activation_fn':None, 'updates_collections':tf.GraphKeys.UPDATE_OPS, 'fused': False}):        
            with tf.variable_scope('Initial'):
                # Transform to channel first for 4D
                if len(x.get_shape().as_list()) == 6:
                    layer = tf.transpose(x,[0,5,1,2,3,4])                      
                    layer = conv4d.conv4d_BatchNorm(input=layer, filters= mdlParams['num_filters_init'], kernel_size=mdlParams['kernel_init'], strides=mdlParams['strides_init'], data_format='channels_first', name='conv1', BATCH = mdlParams['batchSize'], placeholders=placeholders)
                else:
                    layer = slim.convolution(x, mdlParams['num_filters_init'], mdlParams['kernel_init'], stride=mdlParams['strides_init'], scope='conv1')
                # Resnet modules
            with tf.variable_scope('Resnet_modules'):
                # Iterate through all modules
                for i in range(len(mdlParams['ResNet_Size'])):
                    with tf.variable_scope('Module_%d'%(i)):
                        # Iterate through all blocks inside the module
                        for j in range(mdlParams['ResNet_Size'][i]):
                            with tf.variable_scope('Block_%d'%(j)):
                                # Set desired output feature map dimension of the block and the desired stride for the first block in the module
                                if j==0:
                                    output_fm = mdlParams['ResNet_FM'][i]
                                    block_stride = mdlParams['ResNet_Stride'][i]
                                else:
                                    if len(x.get_shape().as_list()) == 6:
                                        block_stride = [1,1,1,1]
                                    else:
                                        block_stride = [1,1,1]
                                if len(x.get_shape().as_list()) == 6:
                                    layer = resneta_block_4d(layer, output_fm, block_stride, mdlParams['batchSize'],placeholders,False,mdlParams['ResNet_Kernels'][i])
                                else:
                                    layer = resneta_block(layer, output_fm, block_stride,kernel_size=mdlParams['ResNet_Kernels'][i],is_temporal=True)
        # GAP
        if mdlParams['use_pooled_features']:
            if len(x.get_shape().as_list()) == 5:
                if tf.__version__ == '1.14.0':
                    layer = math_ops.reduce_mean(layer, axis=[2,3], keepdims = False, name='global_pool')
                else:
                    layer = math_ops.reduce_mean(layer, axis=[2,3], keep_dims = False, name='global_pool')                
            elif len(x.get_shape().as_list()) == 6:
                if tf.__version__ == '1.14.0':
                    layer = math_ops.reduce_mean(layer, axis=[3,4,5], keepdims = False, name='global_pool')
                else:
                    layer = math_ops.reduce_mean(layer, axis=[3,4,5], keep_dims = False, name='global_pool')                
    # Then: features are fed into GRUs
    # Differentiate between pooled and unpooled features
    if mdlParams['use_pooled_features']:
        cnn_features = layer
    else:
        cnn_features = tf.reshape(layer,[mdlParams['batchSize'],mdlParams['timesteps'],layer.get_shape().as_list()[1],layer.get_shape().as_list()[2]])
    # Apply GRUs
    all_cells = [None] * mdlParams['num_GRU_layers']
    for i in range(mdlParams['num_GRU_layers']):
        if mdlParams.get('use_lstm',False):
            all_cells[i] = BN_LSTMCell(mdlParams['num_hidden'],keep_time_dim=mdlParams['keep_time_dim'],is_training=placeholders['train_state'])
        else:
            all_cells[i] = BN_GRUCell(mdlParams['num_hidden'],keep_time_dim=mdlParams['keep_time_dim'],is_training=placeholders['train_state'])
    # Dropout wrapper
        all_cells[i] = rnn.DropoutWrapper(all_cells[i],input_keep_prob=placeholders['KP1'],output_keep_prob=placeholders['KP2'])
    # Stack layers
    gru_cells = rnn.MultiRNNCell(all_cells)
    # Get last output over time from GRU cels
    output_gru, _ = tf.nn.dynamic_rnn(gru_cells, cnn_features, dtype=tf.float32)
    output_gru = tf.transpose(output_gru, [1, 0, 2])
    last = tf.gather(output_gru, int(output_gru.get_shape()[0]) - 1)
    # FC-layer
    output = slim.layers.fully_connected(last, mdlParams['numOut'], activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
    return output    

model_map = {'Resnet4D': Resnet4D,
             'Resnet3D': Resnet3D,
             'convGRUCNN' : convGRUCNN,
             'CNNGRU' : CNNGRU,
               }

def getModel(mdlParams, placeholders):
  """Returns a function for a model
  Args:
    mdlParams: dictionary, contains configuration
    is_training: bool, indicates whether training is active
  Returns:
    model: A function that builds the desired model
  Raises:
    ValueError: If model name is not recognized.
  """
  if mdlParams['model_type'] not in model_map:
    raise ValueError('Name of model unknown %s' % mdlParams['model_type'])
  func = model_map[mdlParams['model_type']]
  @functools.wraps(func)
  def model(x):
      return func(x, mdlParams, placeholders)
  return model