# -*- coding: UTF-8 -*-
import tensorflow as tf


# Credits to: https://github.com/funkey/conv4d
def conv4d(
        input,
        filters,
        kernel_size,
        strides=(1, 1, 1, 1),
        padding='same',
        data_format='channels_last',
        dilation_rate=(1, 1, 1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        trainable=True,
        name=None,
        Stride_Temp=1,
        Stride_Spat=1,
        BATCH=10,
        placeholders =None,
        reuse=None):

    # check arguments
    assert len(input.get_shape().as_list()) == 6, (
        "Tensor of shape (b, c, l, d, h, w) expected")
    assert len(kernel_size) == 4, "4D kernel size expected"

    assert data_format == 'channels_first', (
        "Data format other than 'channels_first' not yet implemented")
    assert dilation_rate == (1, 1, 1, 1), (
        "Dilation rate other than 1 not yet implemented")

    if not name:
        name = 'conv4d'

    # input, kernel, and output sizes
    (b, c_i, l_i, d_i, h_i, w_i) = tuple(input.get_shape().as_list())
    (l_k, d_k, h_k, w_k) = kernel_size
    b=BATCH
    # output size for 'valid' convolution
    if padding == 'valid':
        (l_o, d_o, h_o, w_o) = (
            l_i - l_k + 1,
            d_i - d_k + 1,
            h_i - h_k + 1,
            w_i - w_k + 1
        )
    else:
        (l_o, d_o, h_o, w_o) = (l_i, d_i, h_i, w_i)

    # output tensors for each 3D frame
    
    frame_results = [ None ]*l_o

    # convolve each kernel frame i with each input frame j
    for i in range(l_k):
        # reuse variables of previous 3D convolutions for the same kernel
        # frame (or if the user indicated to have all variables reused)
        reuse_kernel = reuse
        Count = 0

        for j in range(l_i):
            
            # add results to this output frame
            out_frame = j - (i - l_k/2) - (l_i - l_o)/2
            if (out_frame < 0 or out_frame >= l_o):
                continue
            if (Count % Stride_Temp == 0):   #Apply Temporal Stride here
                # convolve input frame j with kernel frame i
                frame_conv3d = tf.layers.conv3d(
                    tf.reshape(input[:,:,j,:], (b, c_i, d_i, h_i, w_i)),
                    filters,
                    kernel_size=(d_k, h_k, w_k),
                    strides=(Stride_Spat, Stride_Spat, Stride_Spat),
                    padding=padding,
                    data_format='channels_first',
                    activation=None,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,
                    trainable=trainable,
                    name=name + '_3dchan%d'%i,
                    reuse=reuse_kernel)

                Count = Count +  1 
                # subsequent frame convolutions should use the same kernel
                reuse_kernel = True
                Frame_Result_check = ((frame_results[int(out_frame)])) #Check if its empty? (For first Iteration)
                # Add Results of Convolutions with the same output frame 
                if Frame_Result_check is None:
                    frame_results[int(out_frame)] = frame_conv3d
                else:
                    frame_results[int(out_frame)] += frame_conv3d
            else:
                Count = Count +  1        

    output = tf.stack(frame_results[0::Stride_Temp], axis=2) #Stack all Resuls into the temporal Dimension (axis=2)

    if activation:
        output = activation(output)

    return output


def conv4d_BatchNorm(
        input,
        filters,
        kernel_size,
        strides=[1, 1, 1, 1],
        padding='same',
        data_format='channels_last',
        dilation_rate=(1, 1, 1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        trainable=True,
        name=None,
        BATCH=20,
        placeholders = None,
        reuse=None):

    # check arguments
    assert len(input.get_shape().as_list()) == 6, (
        "Tensor of shape (b, c, l, d, h, w) expected")
    assert len(kernel_size) == 4, "4D kernel size expected"

    assert data_format == 'channels_first', (
        "Data format other than 'channels_first' not yet implemented")
    assert dilation_rate == (1, 1, 1, 1), (
        "Dilation rate other than 1 not yet implemented")

    if not name:
        name = 'conv4d'
    #print("Input shape",input.get_shape())
    # input, kernel, and output sizes
    (b, c_i, l_i, d_i, h_i, w_i) = tuple(input.get_shape().as_list())
    (l_k, d_k, h_k, w_k) = kernel_size
    b=BATCH
    #print("Input slice",input[:,:,0,:].get_shape())
    #print("First iteration shape",tf.reshape(input[:,:,0,:], (b, c_i, d_i, h_i, w_i)).get_shape())    
    # output size for 'valid' convolution
    if padding == 'valid':
        (l_o, d_o, h_o, w_o) = (
            l_i - l_k + 1,
            d_i - d_k + 1,
            h_i - h_k + 1,
            w_i - w_k + 1
        )
    else:
        (l_o, d_o, h_o, w_o) = (l_i, d_i, h_i, w_i)

    # output tensors for each 3D frame
    
    frame_results = [ None ]*l_o

    # convolve each kernel frame i with each input frame j
    for i in range(l_k):
        # reuse variables of previous 3D convolutions for the same kernel
        # frame (or if the user indicated to have all variables reused)
        reuse_kernel = reuse
        #Count = 0

        for j in range(l_i):
            
            # add results to this output frame
            out_frame = j - (i - l_k/2) - (l_i - l_o)/2
            if (out_frame < 0 or out_frame >= l_o) or (int(out_frame) % strides[0] != 0):
                continue
            #if (Count % strides[0] == 0):   #Apply Temporal Stride here
                # convolve input frame j with kernel frame i
            #print("i",i,"j",j,"out frame",out_frame)
            frame_conv3d = tf.layers.conv3d(
                tf.reshape(input[:,:,j,:], (b, c_i, d_i, h_i, w_i)),
                filters,
                kernel_size=(d_k, h_k, w_k),
                strides=(strides[1], strides[2], strides[3]),
                padding=padding,
                data_format='channels_first',
                activation=None,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                trainable=trainable,
                name=name + '_3dchan%d'%i,
                reuse=reuse_kernel)

            #Count = Count +  1 
            # subsequent frame convolutions should use the same kernel
            reuse_kernel = True
            Frame_Result_check = ((frame_results[int(out_frame)])) #Check if its empty? (For first Iteration)
            # Add Results of Convolutions with the same output frame 
            if Frame_Result_check is None:
                frame_results[int(out_frame)] = frame_conv3d
            else:
                frame_results[int(out_frame)] += frame_conv3d
            #print("fram res", frame_results)
            #else:
            #    Count = Count +  1        

    output = tf.stack(frame_results[0::strides[0]], axis=2) #Stack all Resuls into the temporal Dimension (axis=2)

    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training = placeholders['train_state'],  epsilon=0.0001, decay=0.9, activation_fn=None, data_format='NCHW', updates_collections=tf.GraphKeys.UPDATE_OPS, fused=False)


    if activation:
        output = activation(output)

    return output


def pool4d(
        input,
        kernel_size,
        strides=(1, 1, 1, 1),
        padding='SAME',
        data_format= 'NCDHW',
        name=None,
        BATCH=10,
        placeholders =None):

    # check arguments
    assert len(input.get_shape().as_list()) == 6, (
        "Tensor of shape (b, c, l, d, h, w) expected")
    assert len(kernel_size) == 4, "4D kernel size expected"

    if not name:
        name = 'pool3d'

    # input, kernel, and output sizes
    (b, c_i, l_i, d_i, h_i, w_i) = tuple(input.get_shape().as_list())
    (l_k, d_k, h_k, w_k) = kernel_size
    b=BATCH
    # output size for 'valid' pooling
    if padding == 'valid':
        (l_o, d_o, h_o, w_o) = (
            l_i - l_k + 1,
            d_i - d_k + 1,
            h_i - h_k + 1,
            w_i - w_k + 1
        )
    else:
        (l_o, d_o, h_o, w_o) = (l_i, d_i, h_i, w_i)

    # output tensors for each 3D frame
    
    frame_results = [ None ]*l_o

    # convolve each kernel frame i with each input frame j
    for i in range(l_k):

        Count = 0 # counter for temporal Stride 

        for j in range(l_i):
            
            # add results to this output frame
            out_frame = j - (i - l_k/2) - (l_i - l_o)/2
            if (out_frame < 0 or out_frame >= l_o):
                continue
            if (Count % strides[0] == 0):   #Apply Temporal Stride here
                # convolve input frame j with kernel frame i
                frame_pool3d = tf.nn.pool(
                    tf.reshape(input[:,:,j,:], (b, c_i, d_i, h_i, w_i)),
                    window_shape= (d_k, h_k, w_k),
                    pooling_type = "AVG",
                    padding=padding,
                    strides= (strides[1], strides[2], strides[3]),
                    name= name + '_3dchan%d'%i,
                    data_format='NCDHW')

                Count = Count +  1 
                # subsequent frame convolutions should use the same kernel

                Frame_Result_check = ((frame_results[int(out_frame)])) #Check if its empty? (For first Iteration)
                # Add Results of Convolutions with the same output frame 
                if Frame_Result_check is None:
                    frame_results[int(out_frame)] = frame_pool3d
                else:
                    frame_results[int(out_frame)] += frame_pool3d
            else:
                Count = Count +  1        

    output = tf.stack(frame_results[0::strides[0]], axis=2) #Stack all Resuls into the temporal Dimension (axis=2)
    output = output / l_k #Scale for Average for temporal Window 


    return output


