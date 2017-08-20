import h5py
import numpy as np
import tensorflow as tf
import constants as c
import matplotlib.pyplot as plt

with h5py.File(c.DATA_DIR+'ECOG_40_41_raw.h5', 'r') as h5file:
	with tf.Session() as sess:
	    train_data = np.array(h5file['train'],dtype='float32')
	    resized_data = np.empty([train_data.shape[0],c.TRAIN_HEIGHT*c.TRAIN_WIDTH])
	    frames = train_data.transpose().reshape(c.TRAIN_HEIGHT_RAW,c.TRAIN_WIDTH_RAW,train_data.shape[0])
	    train_label = np.array(h5file['train_label'],dtype='float32')
	    for index in range(0,train_data.shape[0],5000):
	    	print index, '/' , train_data.shape[0]
	    	resized_frames = tf.image.resize_images(frames[:,:,index:min(index+5000,train_data.shape[0])], [c.TRAIN_HEIGHT,c.TRAIN_WIDTH])
	    	imgs = sess.run(resized_frames)

	    	imgs = imgs.reshape(c.TRAIN_HEIGHT*c.TRAIN_WIDTH,imgs.shape[2]).transpose()
	    	resized_data[index:min(index+5000,train_data.shape[0]),:] = imgs
	    
	    

	    valid_data = np.array(h5file['valid'],dtype='float32')
	    resized_data_valid = np.empty([valid_data.shape[0],c.TRAIN_HEIGHT*c.TRAIN_WIDTH])
	    frames = valid_data.transpose().reshape(c.TRAIN_HEIGHT_RAW,c.TRAIN_WIDTH_RAW,valid_data.shape[0])
	    for index in range(0,valid_data.shape[0],5000):
	    	print index, '/' , valid_data.shape[0]
	    	resized_frames = tf.image.resize_images(frames[:,:,index:min(index+5000,valid_data.shape[0])], [c.TRAIN_HEIGHT,c.TRAIN_WIDTH])
	    	imgs = sess.run(resized_frames)

	    	imgs = imgs.reshape(c.TRAIN_HEIGHT*c.TRAIN_WIDTH,imgs.shape[2]).transpose()
	    	resized_data_valid[index:min(index+5000,valid_data.shape[0]),:] = imgs
	    valid_label = np.array(h5file['valid_label'],dtype='float32')

with h5py.File(c.DATA_DIR+'ECOG_40_41.h5','w') as h5file_w:
	h5file_w.create_dataset('train',data = resized_data)
	h5file_w.create_dataset('train_label',data = train_label)
	h5file_w.create_dataset('valid',data = resized_data_valid)
	h5file_w.create_dataset('valid_label',data = valid_label)

