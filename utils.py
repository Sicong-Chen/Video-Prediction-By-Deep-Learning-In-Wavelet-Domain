import tensorflow as tf
import numpy as np
from scipy.ndimage import imread
from scipy.misc import imresize
from glob import glob
from scipy.misc import toimage
import os

import constants as c
from tfutils import log10
import h5py
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import threeDWT as WT

##
# Data
##

def normalize_frames(frames):
    """
    Convert frames from int8 [0, 255] to float32 [-1, 1].

    @param frames: A numpy array. The frames to be converted.

    @return: The normalized frames.
    """
    new_frames = frames.astype(np.float32)
    new_frames /= (255 / 2)
    new_frames -= 1

    return new_frames

def denormalize_frames(frames):
    """
    Performs the inverse operation of normalize_frames.

    @param frames: A numpy array. The frames to be converted.

    @return: The denormalized frames.
    """
    new_frames = frames + 1
    new_frames *= (255 / 2)
    # noinspection PyUnresolvedReferences
    new_frames = new_frames.astype(np.uint8)

    return new_frames

def normalize_clips(clips):
    """
    Convert frames from int8 [0, 255] to float32 [-1, 1].

    @param frames: A numpy array. The frames to be converted.

    @return: The normalized frames.
    """
    new_clips = clips.astype(np.float32)
    new_clips = (new_clips * c.DATA_SCALE_FACTOR - c.DATA_MIN) / (c.DATA_MAX-c.DATA_MIN) * 2 - 1
    new_clips = new_clips.clip(-1,1)
    return new_clips

def denormalize_clips(clips):
    """
    Convert frames from int8 [0, 255] to float32 [-1, 1].

    @param frames: A numpy array. The frames to be converted.

    @return: The normalized frames.
    """
    new_clips = ((clips + 1)/2 * (c.DATA_MAX-c.DATA_MIN) + c.DATA_MIN) / c.DATA_SCALE_FACTOR
    return new_clips

def clip_l2_diff(clip):
    """
    @param clip: A numpy array of shape [c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (3 * (c.HIST_LEN + 1))].
    @return: The sum of l2 differences between the frame pixels of each sequential pair of frames.
    """
    diff = 0
    for i in xrange(c.HIST_LEN):
        frame = clip[:, :, 3 * i:3 * (i + 1)]
        next_frame = clip[:, :, 3 * (i + 1):3 * (i + 2)]
        # noinspection PyTypeChecker
        diff += np.sum(np.square(next_frame - frame))

    return diff

# def get_full_clips(data_dir, num_clips, num_rec_out=1):
#     """
#     Loads a batch of random clips from the unprocessed train or test data.

#     @param data_dir: The directory of the data to read. Should be either c.TRAIN_DIR or c.TEST_DIR.
#     @param num_clips: The number of clips to read.
#     @param num_rec_out: The number of outputs to predict. Outputs > 1 are computed recursively,
#                         using the previously-generated frames as input. Default = 1.

#     @return: An array of shape
#              [num_clips, c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (3 * (c.HIST_LEN + num_rec_out))].
#              A batch of frame sequences with values normalized in range [-1, 1].
#     """
#     clips = np.empty([num_clips,
#                       c.FULL_HEIGHT,
#                       c.FULL_WIDTH,
#                       (3 * (c.HIST_LEN + num_rec_out))])

#     # get num_clips random episodes
#     ep_dirs = np.random.choice(glob(os.path.join(data_dir, '*')), num_clips)

#     # get a random clip of length HIST_LEN + num_rec_out from each episode
#     for clip_num, ep_dir in enumerate(ep_dirs):
#         ep_frame_paths = sorted(glob(os.path.join(ep_dir, '*')))
#         start_index = np.random.choice(len(ep_frame_paths) - (c.HIST_LEN + num_rec_out - 1))
#         clip_frame_paths = ep_frame_paths[start_index:start_index + (c.HIST_LEN + num_rec_out)]

#         # read in frames
#         for frame_num, frame_path in enumerate(clip_frame_paths):
#             frame = imread(frame_path, mode='RGB')
#             norm_frame = normalize_frames(frame)

#             clips[clip_num, :, :, frame_num * 3:(frame_num + 1) * 3] = norm_frame

#     return clips

def get_full_clips(num_clips,train_or_val,num_rec_out=1,Indexs=None):
    """
    Loads a batch of random clips from the unprocessed train or test data.

    @param data_dir: The directory of the data to read. Should be either c.TRAIN_DIR or c.TEST_DIR.
    @param num_clips: The number of clips to read.
    @param num_rec_out: The number of outputs to predict. Outputs > 1 are computed recursively,
                        using the previously-generated frames as input. Default = 1.

    @return: An array of shape
             [num_clips, c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (c.NUM_INPUT_CHANNEL * (c.HIST_LEN + num_rec_out))].
             A batch of frame sequences with values normalized in range [-1, 1].
    """
    clips = np.empty([num_clips,
                      c.FULL_HEIGHT,
                      c.FULL_WIDTH,
                      (c.NUM_INPUT_CHANNEL * (c.HIST_LEN + num_rec_out))])

    WTclips = np.empty([num_clips,
                      c.FULL_HEIGHT,
                      c.FULL_WIDTH,
                      (c.NUM_INPUT_CHANNEL * (c.HIST_LEN + num_rec_out))])

    # get num_clips random episodes
    #ep_dirs = np.random.choice(glob(os.path.join(data_dir, '*')), num_clips)

    with h5py.File(c.DATA_DIR + 'ECOG_40_41.h5', 'r') as h5file:

        for i in xrange(num_clips):

            if train_or_val=='train' or c.RANDON_TEST==1:
                start_index = np.random.choice(h5file[train_or_val].shape[0] - (c.HIST_LEN + num_rec_out - 1))
            else:
                start_index = Indexs[i]
            clip = np.array(h5file[train_or_val][start_index : start_index + (c.HIST_LEN + num_rec_out), :],dtype='float32')

            clips[i] = clip.transpose().reshape(c.TRAIN_HEIGHT,c.TRAIN_WIDTH,c.HIST_LEN + num_rec_out)
            
        #clips = normalize_clips(clips)
        '''
        for i in xrange(num_clips):
            WTclips[i] = WT.threeDWT(clips[i])
        '''
    return clips
    #return WTclips 


def process_clip():
    """
    Gets a clip from the train dataset, cropped randomly to c.TRAIN_HEIGHT x c.TRAIN_WIDTH.

    @return: An array of shape [c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (3 * (c.HIST_LEN + 1))].
             A frame sequence with values normalized in range [-1, 1].
    """
    clip = get_full_clips(c.TRAIN_DIR, 1)[0]

    # Randomly crop the clip. With 0.05 probability, take the first crop offered, otherwise,
    # repeat until we have a clip with movement in it.
    take_first = np.random.choice(2, p=[0.95, 0.05])
    cropped_clip = np.empty([c.TRAIN_HEIGHT, c.TRAIN_WIDTH, 3 * (c.HIST_LEN + 1)])
    for i in xrange(100):  # cap at 100 trials in case the clip has no movement anywhere
        crop_x = np.random.choice(c.FULL_WIDTH - c.TRAIN_WIDTH + 1)
        crop_y = np.random.choice(c.FULL_HEIGHT - c.TRAIN_HEIGHT + 1)
        cropped_clip = clip[crop_y:crop_y + c.TRAIN_HEIGHT, crop_x:crop_x + c.TRAIN_WIDTH, :]

        if take_first or clip_l2_diff(cropped_clip) > c.MOVEMENT_THRESHOLD:
            break

    return cropped_clip

# def get_train_batch():
#     """
#     Loads c.BATCH_SIZE clips from the database of preprocessed training clips.

#     @return: An array of shape
#             [c.BATCH_SIZE, c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (3 * (c.HIST_LEN + 1))].
#     """
#     clips = np.empty([c.BATCH_SIZE, c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (3 * (c.HIST_LEN + 1))],
#                      dtype=np.float32)
#     for i in xrange(c.BATCH_SIZE):
#         path = c.TRAIN_DIR_CLIPS + str(np.random.choice(c.NUM_CLIPS)) + '.npz'
#         clip = np.load(path)['arr_0']

#         clips[i] = clip

#     return clips

def get_train_batch(train_batch_size, num_rec_out=1):
    """
    Loads c.BATCH_SIZE clips from the database of preprocessed training clips.

    @return: An array of shape
            [c.BATCH_SIZE, c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (3 * (c.HIST_LEN + 1))].
    """
    # clips = np.empty([c.BATCH_SIZE, c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (c.NUM_INPUT_CHANNEL * (c.HIST_LEN + 1))],
    #                  dtype=np.float32)
    # for i in xrange(c.BATCH_SIZE):
    #     #path = c.TRAIN_DIR_CLIPS + str(np.random.choice(c.NUM_CLIPS)) + '.npz'
    #     start_index = np.random.choice(h5file['train'].shape[0] - (c.HIST_LEN + num_rec_out - 1))
    #     clip  = np.array(h5file['train'][start_index : start_index + (c.HIST_LEN + num_rec_out), :])
    #     #clip = np.load(path)['arr_0']
    #     clips[i] = clip.transpose().reshape(c.TRAIN_WIDTH,c.TRAIN_HEIGHT,c.HIST_LEN + num_rec_out)

    # return clips
    return get_full_clips(train_batch_size,train_or_val='train', num_rec_out=num_rec_out)


def get_test_batch(test_batch_size, num_rec_out=1, Indexs = None):
    """
    Gets a clip from the test dataset.

    @param test_batch_size: The number of clips.
    @param num_rec_out: The number of outputs to predict. Outputs > 1 are computed recursively,
                        using the previously-generated frames as input. Default = 1.

    @return: An array of shape:
             [test_batch_size, c.TEST_HEIGHT, c.TEST_WIDTH, (3 * (c.HIST_LEN + num_rec_out))].
             A batch of frame sequences with values normalized in range [-1, 1].
    """
    return get_full_clips(test_batch_size,train_or_val='valid', num_rec_out=num_rec_out, Indexs = Indexs)


##
# Error calculation
##

# TODO: Add SSIM error http://www.cns.nyu.edu/pub/eero/wang03-reprint.pdf
# TODO: Unit test error functions.

def psnr_error(gen_frames, gt_frames):
    """
    Computes the Peak Signal to Noise Ratio error between the generated images and the ground
    truth images.

    @param gen_frames: A tensor of shape [batch_size, height, width, 3]. The frames generated by the
                       generator model.
    @param gt_frames: A tensor of shape [batch_size, height, width, 3]. The ground-truth frames for
                      each frame in gen_frames.

    @return: A scalar tensor. The mean Peak Signal to Noise Ratio error over each frame in the
             batch.
    """
    shape = tf.shape(gen_frames)
    frame_num_pixels = tf.to_float(shape[1] * shape[2])
    num_pixels = tf.to_float(shape[1] * shape[2] * shape[3])
    num_pixels_frame = tf.to_float(shape[1] * shape[2])

    square_diff = tf.square(gt_frames - gen_frames)

    # batch_errors = 10 * log10(1 / ((1 / num_pixels) * tf.reduce_sum(square_diff, [1, 2, 3])))
    # batch_errors_frame = 10 * log10(1 / ((1 / num_pixels_frame) * tf.reduce_sum(square_diff, [1, 2])))
    batch_errors_frame = 10 * log10(4 / ((1 / num_pixels_frame) * tf.reduce_sum(square_diff, [1, 2])))
    batch_errors = tf.reduce_mean(batch_errors_frame,[1])
    # batch_errors = tf.reduce_mean(square_diff, [1, 2, 3])
    # batch_errors_frame = tf.reduce_mean(square_diff, [1, 2])
    return tf.reduce_mean(batch_errors), batch_errors, tf.reduce_mean(batch_errors_frame,0)

def sharp_diff_error(gen_frames, gt_frames):
    """
    Computes the Sharpness Difference error between the generated images and the ground truth
    images.

    @param gen_frames: A tensor of shape [batch_size, height, width, 3]. The frames generated by the
                       generator model.
    @param gt_frames: A tensor of shape [batch_size, height, width, 3]. The ground-truth frames for
                      each frame in gen_frames.

    @return: A scalar tensor. The Sharpness Difference error over each frame in the batch.
    """
    shape = tf.shape(gen_frames)
    num_pixels = tf.to_float(shape[1] * shape[2] * shape[3])

    # gradient difference
    # create filters [-1, 1] and [[1],[-1]] for diffing to the left and down respectively.
    # TODO: Could this be simplified with one filter [[-1, 2], [0, -1]]?
    pos = tf.constant(np.identity(c.NUM_INPUT_CHANNEL*c.PRED_LEN), dtype=tf.float32)
    neg = -1 * pos
    filter_x = tf.expand_dims(tf.pack([neg, pos]), 0)  # [-1, 1]
    filter_y = tf.pack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])  # [[1],[-1]]
    strides = [1, 1, 1, 1]  # stride of (1, 1)
    padding = 'SAME'

    gen_dx = tf.abs(tf.nn.conv2d(gen_frames, filter_x, strides, padding=padding))
    gen_dy = tf.abs(tf.nn.conv2d(gen_frames, filter_y, strides, padding=padding))
    gt_dx = tf.abs(tf.nn.conv2d(gt_frames, filter_x, strides, padding=padding))
    gt_dy = tf.abs(tf.nn.conv2d(gt_frames, filter_y, strides, padding=padding))

    gen_grad_sum = gen_dx + gen_dy
    gt_grad_sum = gt_dx + gt_dy

    grad_diff = tf.abs(gt_grad_sum - gen_grad_sum)

    batch_errors = 10 * log10(1 / ((1 / num_pixels) * tf.reduce_sum(grad_diff, [1, 2, 3])))
    return tf.reduce_mean(batch_errors)

def pseudocolor(val, minval, maxval):
    # Scale val to be in the range [0, 1]
    val = (val - minval) / (maxval - minval)
    # Return RGBA tuple from jet colormap
    return cm.jet(val)[:,:,:,0:3]

def display_result(input_frames,gen_img,gt_img,output_file=None, Save_or_Show = 'save'):
    marge = 6
    #input_frames = np.squeeze(input_frames)
    input_frames = (input_frames+1)/2
    #gen_img = np.squeeze(gen_img)
    gen_img[gen_img>1.] = 1.
    gen_img[gen_img<-1.] = -1.
    gen_img = (gen_img+1)/2
    #gt_img = np.squeeze(gt_img)
    gt_img = (gt_img+1)/2

    len_input = input_frames.shape[2]

    len_gen = gen_img.shape[2]
    len_total = len_input+len_gen

    image_measure = [c.FULL_HEIGHT, c.FULL_WIDTH]
    canv_size = [4*marge+3*c.FULL_HEIGHT , (len_total+1)*marge+len_total*c.FULL_WIDTH, 3]
    future_bengin = len_input*(marge+image_measure[1])
    gt_bengin = marge+image_measure[0]
    error_bengin = 2*(marge+image_measure[0])

    error = np.abs(gt_img-gen_img)
    error = pseudocolor(error, -0.4, 0.4)
    error0 = np.empty([c.FULL_HEIGHT, c.FULL_WIDTH, len_gen,3],dtype=np.float32)
    for i in xrange(len_gen):
        error0[:,:,i,:] = imresize((error[:,:,i,:]*255).astype('uint8'),[c.FULL_HEIGHT, c.FULL_WIDTH,3],interp='nearest')



    input_frames = pseudocolor(input_frames, 0.3, 0.8)
    input_frames0 = np.empty([c.FULL_HEIGHT, c.FULL_WIDTH, len_input,3],dtype=np.float32)
    for i in xrange(len_input):
        input_frames0[:,:,i,:] = imresize((input_frames[:,:,i,:]*255).astype('uint8'),[c.FULL_HEIGHT, c.FULL_WIDTH,3],interp='nearest')


    gen_img = pseudocolor(gen_img, 0.3, 0.8)
    gen_img0 = np.empty([c.FULL_HEIGHT, c.FULL_WIDTH, len_gen,3],dtype=np.float32)
    for i in xrange(len_gen):
        gen_img0[:,:,i,:] = imresize((gen_img[:,:,i,:]*255).astype('uint8'),[c.FULL_HEIGHT, c.FULL_WIDTH,3],interp='nearest')


    gt_img = pseudocolor(gt_img, 0.3, 0.8)
    gt_img0 = np.empty([c.FULL_HEIGHT, c.FULL_WIDTH, len_gen,3],dtype=np.float32)
    for i in xrange(len_gen):
        gt_img0[:,:,i,:] = imresize((gt_img[:,:,i,:]*255).astype('uint8'),[c.FULL_HEIGHT, c.FULL_WIDTH,3],interp='nearest')



    canvas = np.ones(canv_size)*255

    #plt.figure(1, figsize=(len_total, 3))


    for i in xrange(len_input):
        canvas[marge:marge+image_measure[0], marge+i*(marge+image_measure[1]):marge+i*(marge+image_measure[1])+image_measure[1], :] = input_frames0[:,:,i,:]


    for i in xrange(len_gen):

        canvas[marge : marge+image_measure[0],\
                marge+future_bengin+i*(marge+image_measure[1]) : marge+future_bengin+i*(marge+image_measure[1])+image_measure[1], :] = gen_img0[:,:,i,:]

    for i in xrange(len_gen):
        canvas[gt_bengin+marge : gt_bengin+marge+image_measure[0], \
               marge+future_bengin+i*(marge+image_measure[1]) : marge+future_bengin+i*(marge+image_measure[1])+image_measure[1], :] = gt_img0[:,:,i,:]

    for i in xrange(len_gen):
        canvas[error_bengin+marge : error_bengin+marge+image_measure[0], \
               marge+future_bengin+i*(marge+image_measure[1]) : marge+future_bengin+i*(marge+image_measure[1])+image_measure[1], :] = error0[:,:,i,:]

    if Save_or_Show=='save':
        toimage(canvas,cmin=0,cmax=255).save(output_file, quality=100)
    if Save_or_Show=='show':
        imgplt = plt.imshow(canvas)
        plt.show()
