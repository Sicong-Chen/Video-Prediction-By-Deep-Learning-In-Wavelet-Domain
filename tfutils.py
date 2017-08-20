import tensorflow as tf
import numpy as np


def w(shape, stddev=0.01, scope = 'Weight_block'):
    """
    @return A weight layer with the given shape and standard deviation. Initialized with a
            truncated normal distribution.
    """
    with tf.variable_scope(scope):
        return tf.get_variable(name = 'Weight_block', shape = shape, initializer = tf.truncated_normal_initializer(stddev=stddev))


def b(shape, const=0.1, scope = 'Bias_block'):
    """
    @return A bias layer with the given shape.
    """
    with tf.variable_scope(scope):
        return tf.get_variable(name='Bias_block', shape=shape, initializer = tf.constant_initializer(const))

def beta_(shape, const=0.0, scope = 'beta'):
    with tf.variable_scope(scope):
            return tf.get_variable(name='beta', shape=shape, initializer=tf.constant_initializer(const), 
                                    trainable=True)

def gamma_(shape, mean=1.0, stddev=0.02,  scope = 'gamma'):
    with tf.variable_scope(scope):
        return tf.get_variable(name='gamma', shape=shape, initializer=tf.random_normal_initializer(mean, stddev),
                                trainable=True)

def pop_mean_(shape, const=0.0,  scope = 'pop_mean'):
    with tf.variable_scope(scope):
        return tf.get_variable(name='pop_mean', shape=shape, initializer=tf.constant_initializer(const), 
                                trainable=False)

def pop_var_(shape, const=1.0,  scope = 'pop_var'):
    with tf.variable_scope(scope):     
        return tf.get_variable(name='pop_var', shape=shape, initializer=tf.constant_initializer(const), 
                                trainable=False)
# def batch_norm(x, n_out, phase_train, scope='bn', decay=0.999, eps=1e-5, stddev=0.02):
#     """
#     perform 3d convolution batch normalization
#     """
#     #with tf.variable_scope(scope):
#     # beta = tf.get_variable(name='beta', shape=[n_out], initializer=tf.constant_initializer(0.0),
#     #                         trainable=True)
#     # gamma = tf.get_variable(name='gamma', shape=[n_out], initializer=tf.random_normal_initializer(1.0, stddev),
#     #                         trainable=True)
#     beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
#     gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)

#     dim = len(x.get_shape().as_list())
#     axs = [i for i in xrange(dim-1)]
#     batch_mean, batch_var = tf.nn.moments(x, axs, name='moments')

#     ema = tf.train.ExponentialMovingAverage(decay=decay)

#     def mean_var_with_update():
#         ema_apply_op = ema.apply([batch_mean, batch_var])
#         with tf.control_dependencies([ema_apply_op]):
#             return tf.identity(batch_mean), tf.identity(batch_var)

#     mean, var = tf.cond(phase_train,
#                         mean_var_with_update,
#                         lambda: (ema.average(batch_mean), ema.average(batch_var)))
#     normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)


#     return normed


def batch_norm(x, beta, gamma, pop_mean, pop_var , phase_train, decay=0.999, eps=1e-5):
    """
    perform 3d convolution batch normalization
    """

    # with tf.variable_scope(scope):
    #     beta = tf.get_variable(name='beta', shape=[n_out], initializer=tf.constant_initializer(0.0), 
    #                             trainable=True)
    #     gamma = tf.get_variable(name='gamma', shape=[n_out], initializer=tf.random_normal_initializer(1.0, stddev),
    #                             trainable=True)
    #     pop_mean = tf.get_variable(name='pop_mean', shape=[x.get_shape()[-1]], initializer=tf.constant_initializer(0.0), 
    #                             trainable=False)
    #     pop_var = tf.get_variable(name='pop_var', shape=[x.get_shape()[-1]], initializer=tf.constant_initializer(1.0), 
    #                             trainable=False)

    dim = len(x.get_shape().as_list())
    axs = [i for i in xrange(dim-1)]
    batch_mean, batch_var = tf.nn.moments(x, axs, name='moments')

    def mean_var_with_update():
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean,train_var]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(phase_train,
                        mean_var_with_update,
                        lambda: (pop_mean, pop_var) )
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)


    return normed

def conv_out_size(i, p, k, s):
    """
    Gets the output size for a 2D convolution. (Assumes square input and kernel).

    @param i: The side length of the input.
    @param p: The padding type (either 'SAME' or 'VALID').
    @param k: The side length of the kernel.
    @param s: The stride.

    @type i: int
    @type p: string
    @type k: int
    @type s: int

    @return The side length of the output.
    """
    # convert p to a number
    if p == 'SAME':
        p = k // 2
    elif p == 'VALID':
        p = 0
    else:
        raise ValueError('p must be "SAME" or "VALID".')

    return int(((i + (2 * p) - k) / s) + 1)


def log10(t):
    """
    Calculates the base-10 log of each element in t.

    @param t: The tensor from which to calculate the base-10 log.

    @return: A tensor with the base-10 log of each element in t.
    """

    numerator = tf.log(t)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def batch_pad_to_bounding_box(images, offset_height, offset_width, target_height, target_width):
    """
    Zero-pads a batch of images with the given dimensions.

    @param images: 4-D tensor with shape [batch_size, height, width, channels]
    @param offset_height: Number of rows of zeros to add on top.
    @param offset_width: Number of columns of zeros to add on the left.
    @param target_height: Height of output images.
    @param target_width: Width of output images.

    @return: The batch of images, all zero-padded with the specified dimensions.
    """
    batch_size, height, width, channels = tf.Session().run(tf.shape(images))

    if not offset_height >= 0:
        raise ValueError('offset_height must be >= 0')
    if not offset_width >= 0:
        raise ValueError('offset_width must be >= 0')
    if not target_height >= height + offset_height:
        raise ValueError('target_height must be >= height + offset_height')
    if not target_width >= width + offset_width:
        raise ValueError('target_width must be >= width + offset_width')

    num_tpad = offset_height
    num_lpad = offset_width
    num_bpad = target_height - (height + offset_height)
    num_rpad = target_width - (width + offset_width)

    tpad = np.zeros([batch_size, num_tpad, width, channels])
    bpad = np.zeros([batch_size, num_bpad, width, channels])
    lpad = np.zeros([batch_size, target_height, num_lpad, channels])
    rpad = np.zeros([batch_size, target_height, num_rpad, channels])

    padded = images
    if num_tpad > 0 and num_bpad > 0: padded = tf.concat(1, [tpad, padded, bpad])
    elif num_tpad > 0: padded = tf.concat(1, [tpad, padded])
    elif num_bpad > 0: padded = tf.concat(1, [padded, bpad])
    if num_lpad > 0 and num_rpad > 0: padded = tf.concat(2, [lpad, padded, rpad])
    elif num_lpad > 0: padded = tf.concat(2, [lpad, padded])
    elif num_rpad > 0: padded = tf.concat(2, [padded, rpad])

    return padded


def batch_crop_to_bounding_box(images, offset_height, offset_width, target_height, target_width):
    """
    Crops a batch of images to the given dimensions.

    @param images: 4-D tensor with shape [batch, height, width, channels]
    @param offset_height: Vertical coordinate of the top-left corner of the result in the input.
    @param offset_width: Horizontal coordinate of the top-left corner of the result in the input.
    @param target_height: Height of output images.
    @param target_width: Width of output images.

    @return: The batch of images, all cropped the specified dimensions.
    """
    batch_size, height, width, channels = tf.Session().run(tf.shape(images))

    if not offset_height >= 0:
        raise ValueError('offset_height must be >= 0')
    if not offset_width >= 0:
        raise ValueError('offset_width must be >= 0')
    if not target_height + offset_height <= height:
        raise ValueError('target_height + offset_height must be <= height')
    if not target_width <= width - offset_width:
        raise ValueError('target_width + offset_width must be <= width')

    top = offset_height
    bottom = target_height + offset_height
    left = offset_width
    right = target_width + offset_width

    return images[:, top:bottom, left:right, :]

def video_downsample(video,factor):
    
    output = video[:,:,:,0::int(factor)]

    for i in xrange(1,int(factor)):
        output += video[:,:,:,i::int(factor)]
    output /= factor
    return output

def video_upsample(video,factor=2):
    imsize =video.get_shape().as_list()
    
    output_list = []
    for f in xrange(int(imsize[3])):
        output_list.append(video[:,:,:,f])
        output_list.append(video[:,:,:,f])
    
    output = tf.pack(output_list,axis=3)
    return output