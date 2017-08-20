import tensorflow as tf
import numpy as np
import re
from skimage.transform import resize

from d_scale_model import DScaleModel
import constants as c
if c.WGAN:
    from loss_functions_WGAN import d_loss
else:
    from loss_functions import combined_loss, adv_loss

from tfutils import video_downsample


# noinspection PyShadowingNames
class DiscriminatorModel:
    def __init__(self, session, summary_writer, height, width, scale_conv_layer_fms,
                 scale_kernel_sizes, scale_fc_layer_sizes):
        """
        Initializes a DiscriminatorModel.

        @param session: The TensorFlow session.
        @param summary_writer: The writer object to record TensorBoard summaries
        @param height: The height of the input images.
        @param width: The width of the input images.
        @param scale_conv_layer_fms: The number of feature maps in each convolutional layer of each
                                     scale network.
        @param scale_kernel_sizes: The size of the kernel for each layer of each scale network.
        @param scale_fc_layer_sizes: The number of nodes in each fully-connected layer of each scale
                                     network.

        @type session: tf.Session
        @type summary_writer: tf.train.SummaryWriter
        @type height: int
        @type width: int
        @type scale_conv_layer_fms: list<list<int>>
        @type scale_kernel_sizes: list<list<int>>
        @type scale_fc_layer_sizes: list<list<int>>
        """
        self.sess = session
        self.summary_writer = summary_writer
        self.height = height
        self.width = width
        self.scale_conv_layer_fms = scale_conv_layer_fms
        self.scale_kernel_sizes = scale_kernel_sizes
        self.scale_fc_layer_sizes = scale_fc_layer_sizes
        self.num_scale_nets = len(scale_conv_layer_fms)
        self.teach = 0

        self.define_graph()

    # noinspection PyAttributeOutsideInit
    def define_graph(self):
        """
        Sets up the model graph in TensorFlow.
        """
        with tf.variable_scope('discriminator') as scope:
            ##
            # Setup scale networks. Each will make the predictions for images at a given scale.
            ##
            # net for fake samples
            self.scale_nets = []

            for scale_num in xrange(self.num_scale_nets):
                with tf.variable_scope('scale_net_' + str(scale_num)):
                    scale_factor = 1. / 2 ** ((self.num_scale_nets - 1) - scale_num)

                    self.scale_nets.append(DScaleModel(scale_num,
                                                       int(self.height * scale_factor),
                                                       int(self.width * scale_factor),
                                                       self.scale_conv_layer_fms[scale_num],
                                                       self.scale_kernel_sizes[scale_num],
                                                       self.scale_fc_layer_sizes[scale_num]))

            # A list of the prediction tensors for each scale network
            self.scale_preds = []
            for scale_num in xrange(self.num_scale_nets):
                self.scale_preds.append(self.scale_nets[scale_num].preds)


            scope.reuse_variables()


            # net for real samples
            self.scale_nets_real = []

            for scale_num in xrange(self.num_scale_nets):
                with tf.variable_scope('scale_net_' + str(scale_num)):
                    scale_factor = 1. / 2 ** ((self.num_scale_nets - 1) - scale_num)

                    self.scale_nets_real.append(DScaleModel(scale_num,
                                                       int(self.height * scale_factor),
                                                       int(self.width * scale_factor),
                                                       self.scale_conv_layer_fms[scale_num],
                                                       self.scale_kernel_sizes[scale_num],
                                                       self.scale_fc_layer_sizes[scale_num]))

            # A list of the prediction tensors for each scale network
            self.scale_preds_real = []
            for scale_num in xrange(self.num_scale_nets):
                self.scale_preds_real.append(self.scale_nets_real[scale_num].preds)

            with tf.name_scope('training'):
                # For Non-teacher-forcing
                ## global loss is the combined loss from every scale network
                self.d_loss = d_loss(self.scale_preds, self.scale_preds_real)[0]

                #self.d_loss = d_loss(self.scale_preds, self.scale_preds)[0]
                self.global_step = tf.Variable(0, trainable=False, name='global_step')
                #self.optimizer = tf.train.AdamOptimizer(learning_rate=c.LRATE_D, name='optimizer')#tf.train.GradientDescentOptimizer(c.LRATE_D, name='optimizer')#tf.train.GradientDescentOptimizer(c.LRATE_D, name='optimizer')#tf.train.AdamOptimizer(learning_rate=c.LRATE_D, name='optimizer')#tf.train.GradientDescentOptimizer(c.LRATE_D, name='optimizer')
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=c.LRATE_D, name='optimizer')

                self.train_op = self.optimizer.minimize(self.d_loss,
                                                        global_step=self.global_step,
                                                        name='train_op')
                self.theta_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')


                self.cc = []
                for item in self.theta_c:
                    if re.search('Weight_block',item.name) is not None:
                        self.cc.append(item)

                
                clipped_var_c = [tf.assign(var, tf.clip_by_value(var, c.WGAN_CLAMP_LOWER, c.WGAN_CLAMP_UPPER)) for var in self.cc]
                with tf.control_dependencies([self.train_op]):
                    self.train_op = tf.tuple(clipped_var_c)

                # add summaries to visualize in TensorBoard
                loss_summary = tf.scalar_summary('loss_D', self.d_loss)
                self.summaries = tf.merge_summary([loss_summary])


    def build_feed_dict(self, input_frames, gt_output_frames, generator):
        """
        Builds a feed_dict with resized inputs and outputs for each scale network.

        @param input_frames: An array of shape
                             [batch_size x self.height x self.width x (3 * HIST_LEN)], The frames to
                             use for generation.
        @param gt_output_frames: An array of shape [batch_size x self.height x self.width x 3], The
                                 ground truth outputs for each sequence in input_frames.
        @param generator: The generator model.

        @return: The feed_dict needed to run this network, all scale_nets, and the generator
                 predictions.
        """
        feed_dict = {}
        
        batch_size = np.shape(gt_output_frames)[0]

        ##
        # Get generated frames from GeneratorModel
        ##
        
        g_feed_dict = {generator.input_frames_train: input_frames,
                       generator.gt_frames_train: gt_output_frames,
                       generator.convKeepProb: 1,#c.CONV_KEEPPROB,
                       generator.teacher_forcing: self.teach,
                       generator.bn_mode: False}
                       #generator.convKeepProb: 1.0}
        g_scale_preds = self.sess.run(generator.scale_preds_train, feed_dict=g_feed_dict)


        

        ##
        # Create discriminator feed dict
        ##
        for scale_num in xrange(self.num_scale_nets):
            scale_net = self.scale_nets[scale_num]
            scale_net_real = self.scale_nets_real[scale_num]
            scale_factor = 1. / 2 ** ((self.num_scale_nets - 1) - scale_num)
            # resize gt_output_frames
            scaled_gt_output_frames = np.empty([batch_size, scale_net.height, scale_net.width, c.NUM_INPUT_CHANNEL*c.PRED_LEN])
            for i, img in enumerate(gt_output_frames):
                # for skimage.transform.resize, images need to be in range [0, 1], so normalize to
                # [0, 1] before resize and back to [-1, 1] after
                sknorm_img = (img / 2) + 0.5
                resized_frame = resize(sknorm_img, [scale_net.height, scale_net.width, c.NUM_INPUT_CHANNEL*c.PRED_LEN])
                scaled_gt_output_frames[i] = (resized_frame - 0.5) * 2
            if c.DOTEMPDOWN is True:
                scaled_gt_output_frames = video_downsample(scaled_gt_output_frames,1/scale_factor)
            # resize hist_frames 
            scaled_hist_frames = np.empty([batch_size, scale_net.height, scale_net.width, c.NUM_INPUT_CHANNEL*c.HIST_LEN])
            for i, img in enumerate(input_frames):
                # for skimage.transform.resize, images need to be in range [0, 1], so normalize to
                # [0, 1] before resize and back to [-1, 1] after
                sknorm_img = (img / 2) + 0.5
                resized_frame = resize(sknorm_img, [scale_net.height, scale_net.width, c.NUM_INPUT_CHANNEL*c.HIST_LEN])
                scaled_hist_frames[i] = (resized_frame - 0.5) * 2
            if c.DOTEMPDOWN is True:
                scaled_hist_frames = video_downsample(scaled_hist_frames,1/scale_factor)
            
            # combine with resized gt_output_frames to get inputs for prediction
            if c.CONSIDER_PAST_FRAMES == 1:
                scaled_all_frames_g = np.concatenate([scaled_hist_frames, g_scale_preds[scale_num]],axis=3)
                scaled_all_frames_gt = np.concatenate([scaled_hist_frames, scaled_gt_output_frames],axis=3)
                scaled_input_frames = np.concatenate([scaled_all_frames_g, scaled_all_frames_gt])
            if c.CONSIDER_PAST_FRAMES == 0:
                scaled_input_frames = np.concatenate([g_scale_preds[scale_num],
                                                      scaled_gt_output_frames])

            # convert to np array and add to feed_dict
            feed_dict[scale_net.input_frames] = scaled_all_frames_g
            feed_dict[scale_net.fcKeepProb] = c.FC_KEEPPROB
            feed_dict[scale_net.convKeepProb] = c.CONV_KEEPPROB
            feed_dict[scale_net.bn_mode] = True

            feed_dict[scale_net_real.input_frames] = scaled_all_frames_gt
            feed_dict[scale_net_real.fcKeepProb] = c.FC_KEEPPROB
            feed_dict[scale_net_real.convKeepProb] = c.CONV_KEEPPROB
            feed_dict[scale_net_real.bn_mode] = True


        # add labels for each image to feed_dict

        return feed_dict

    def train_step(self, batch, generator, teach = 0):
        """
        Runs a training step using the global loss on each of the scale networks.

        @param batch: An array of shape
                      [BATCH_SIZE x self.height x self.width x (3 * (HIST_LEN + 1))]. The input
                      and output frames, concatenated along the channel axis (index 3).
        @param generator: The generator model.

        @return: The global step.
        """
        ##
        # Split into inputs and outputs
        ##
        self.teach = teach
        if c.CONSIDER_PAST_FRAMES == 0:
            input_frames = batch[:, :, :, :-c.NUM_INPUT_CHANNEL*c.PRED_LEN]
            gt_output_frames = batch[:, :, :, -c.NUM_INPUT_CHANNEL*c.PRED_LEN:]

        if c.CONSIDER_PAST_FRAMES == 1:
            #input_frames = batch[:, :, :, :]
            input_frames = batch[:, :, :, :-c.NUM_INPUT_CHANNEL*c.PRED_LEN]
            gt_output_frames = batch[:, :, :, -c.NUM_INPUT_CHANNEL*c.PRED_LEN:]

        ##
        # Resize inputs and gt_frames to pseudo_size
        ##
        #input_frames = tf.image.resize_images(input_frames,[c.PSEUDO_HEIGHT,c.PSEUDO_WIDTH])
        #gt_output_frames = tf.image.resize_images(gt_output_frames,[c.PSEUDO_HEIGHT,c.PSEUDO_WIDTH])

        ##
        # Train
        ##

        global_step = self.sess.run(self.global_step)
        # if global_step < 2500 or global_step % 500 == 0:
        #     citers = 100
        # else:
        #     citers = c.WGAN_CITERS
        citers = c.WGAN_CITERS
        for _ in xrange(citers):
            feed_dict = self.build_feed_dict(input_frames, gt_output_frames, generator)
            
            # if c.TEACTHER_FORCE == 0:
            _, d_loss, global_step, summaries = self.sess.run(
                [self.train_op, self.d_loss, self.global_step, self.summaries],
                feed_dict=feed_dict)

            
            #fake,real = self.sess.run([self.scale_preds,self.scale_preds_real],feed_dict=feed_dict) 


        # else:
        #     _, d_loss_scale, global_step, summaries = self.sess.run(
        #         [self.train_op_scale, self.d_loss_scale, self.global_step, self.summaries],
        #         feed_dict=feed_dict)

        ##
        # User output
        ##

        if global_step % c.STATS_FREQ == 0:
            print 'DiscriminatorModel: step %d | global loss: %f' % (global_step, d_loss)
        if global_step % c.SUMMARY_FREQ == 0:
            print 'DiscriminatorModel: saved summaries'
            self.summary_writer.add_summary(summaries, global_step)

        return global_step
