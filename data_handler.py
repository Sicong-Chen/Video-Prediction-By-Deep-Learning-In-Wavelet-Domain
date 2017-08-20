import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ion()

import cPickle as pkl
import sys
import pdb
import h5py
class VideoPatchDataHandler(object):
  def __init__(self,sequence_length=20,batch_size=80,down_sample_rate_=1,dataset_name='train'):
    stats = pkl.load(open('/scratch/ys1297/ecog/data/ECOG_4041_mean.pkl','r'))
    self.data_file_ = h5py.File('/scratch/ys1297/ecog/data/ECOG_40_41.h5','r')[dataset_name]
    # sample the data base on down_sample_rate
    self.down_sample_rate_ = down_sample_rate_
    self.seq_length_ = sequence_length
    self.batch_size_ = batch_size
    self.image_size_ = [32,32]#[20,18]
   
     
    try:
      self.data_ = np.float32(self.data_file_[:]*stats['scale_factor']) #data 36 min:-120 max:103
      self.data_ = (self.data_-stats['min'])/(stats['max']-stats['min'])
      #self.data_ = np.float32(self.data_file_[:]*10000) #data 41
      #self.data_ = (self.data_+10)/(20+10)
        # self.data_ min: -150 max: 103
      self.data_[self.data_<0]=0
      self.data_[self.data_>1]=1
    except:
      print 'Please set the correct path to the dataset'
      sys.exit()
    self.dataset_size_ = self.data_.shape[0]
    self.id_= 0
    self.indices_ = np.arange((self.dataset_size_-self.seq_length_)/self.down_sample_rate_)
    self.is_color_ = False
  def GetBatch(self, verbose=False):
    minibatch = np.zeros((self.batch_size_, self.seq_length_, self.image_size_[0],self.image_size_[1],1), dtype=np.float32)

    for j in range(self.batch_size_):
      temp =  self.data_[self.indices_[self.id_]*self.down_sample_rate_:self.indices_[self.id_]*self.down_sample_rate_+self.seq_length_,:]
      minibatch[j,:,:,:,0]  = temp.reshape(-1,32,32)#(-1,20,18)
      self.id_ += 1
      if self.id_ >= (self.dataset_size_-self.seq_length_-1)/self.down_sample_rate_:
        self.id_ = 0
        self.indices_ = np.arange((self.dataset_size_-self.seq_length_)/self.down_sample_rate_)
        np.random.shuffle(self.indices_)

    return minibatch
  
  def Set_id(self,id):
    self.id_ = id
    pass
  
  def Get_ordered_Batch(self):
    minibatch = np.zeros((self.batch_size_, self.seq_length_, self.image_size_[0],self.image_size_[1],1), dtype=np.float32)

    for j in range(self.batch_size_):  
      temp =  self.data_[self.indices_[self.id_]*self.down_sample_rate_:self.indices_[self.id_]*self.down_sample_rate_+self.seq_length_,
:]
      minibatch[j,:,:,:,0]  = temp.reshape(-1,32,32)#(-1,20,18)
      self.id_ += 1
      if self.id_ >= (self.dataset_size_-self.seq_length_-1)/self.down_sample_rate_:
        self.id_ = 0
        self.indices_ = np.arange((self.dataset_size_-self.seq_length_)/self.down_sample_rate_)
        np.random.shuffle(self.indices_)

    return minibatch
  def DisplayData_Ecog(self, data, rec=None, fut=None, fig=1, case_id=0, output_file=None):
    output_file1 = None
    output_file2 = None

    if output_file is not None:
      name, ext = os.path.splitext(output_file)
      output_file1 = '%s_original%s' % (name, ext)
      output_file2 = '%s_recon%s' % (name, ext)
      output_file3 = '%s_original_norm_color%s' % (name, ext)
    # get data
    if self.is_color_:
      data = data[case_id, :]
      data[data>1.] = 1.
      data[data<0.] = 0.
      data = data.reshape(-1, 3, self.image_size_, self.image_size_)
      data = data.transpose(0, 2, 3, 1)
    else:
      #data = data[case_id, :].reshape(-1, 20, 18)
      data = np.squeeze(data[case_id,:,:,:,0])
      data = np.swapaxes(data,1,2)

    # get reconstruction and future sequences if they exist
    if rec is not None:
      if self.is_color_:
        rec = rec[case_id, :]
        rec[rec>1.] = 1.
        rec[rec<0.] = 0.
        rec = rec.reshape(-1, 3, self.image_size_, self.image_size_)
        rec = rec.transpose(0, 2, 3, 1)
      else:
        #rec = rec[case_id, :].reshape(-1, 20,18)
        rec = np.squeeze(rec[case_id,:,:,:,0]) 
	rec = np.swapaxes(rec,1,2)
      enc_seq_length = rec.shape[0]
    else:
      rec = data[0:10,:,:]
      enc_seq_length = rec.shape[0]	
    
    if fut is not None:
      if self.is_color_:
        fut = fut[case_id, :]
        fut[fut>1.] = 1.
        fut[fut<0.] = 0.
        fut = fut.reshape(-1, 3, self.image_size_, self.image_size_)
        fut = fut.transpose(0, 2, 3, 1)
      else:
        #fut = fut[case_id, :].reshape(-1, 20,18)
        fut = np.squeeze(fut[case_id,:,:,:,0])
	fut = np.swapaxes(fut,1,2)
    else:
      fut = data[9:-1,:,:]

    if rec is None:
      enc_seq_length = self.seq_length_ - fut.shape[0]
    else:
      assert enc_seq_length == self.seq_length_ - fut.shape[0]

    """ new plot method """
    num_rows = 3
    print "here"
    plt.figure(2*fig, figsize=(self.seq_length_, 3))
    plt.clf()
    #plt.figure(1)
    for i in xrange(self.seq_length_):
      plt.subplot(num_rows, self.seq_length_, i+1)
      if self.is_color_:
        plt.imshow(data[i])
      else:
        plt.imshow(-data[i, :, :], cmap=plt.cm.jet, interpolation="nearest")
        plt.clim(-0.8,-0.3)
      plt.axis('off')

    # create figure for reconstuction and future sequences
    for i in xrange(self.seq_length_):
      if rec is not None and i < enc_seq_length:
        plt.subplot(num_rows, self.seq_length_, i + 1 + self.seq_length_)
        if self.is_color_:
          plt.imshow(rec[rec.shape[0] - i - 1])
        else:
          plt.imshow(-rec[rec.shape[0] - i - 1, :, :], cmap=plt.cm.jet , interpolation="nearest")
          plt.clim(-0.8,-0.3)
      if fut is not None and i >= enc_seq_length:
        plt.subplot(num_rows, self.seq_length_, i + 1+self.seq_length_)
        if self.is_color_:
          plt.imshow(fut[i - enc_seq_length])
        else:
          plt.imshow(-fut[i - enc_seq_length, :, :], cmap=plt.cm.jet,  interpolation="nearest")
          plt.clim(-0.8,-0.3)
      plt.axis('off')

    error_sum = 0
    for i in xrange(self.seq_length_):
      if rec is not None and i < enc_seq_length:
        plt.subplot(num_rows, self.seq_length_, i + 1 + self.seq_length_*2)
        if self.is_color_:
          plt.imshow(rec[rec.shape[0] - i - 1])
        else:
          plt.imshow(-data[i, :, :]+rec[rec.shape[0] - i - 1, :, :], cmap=plt.cm.jet , interpolation="nearest")
          plt.clim(-0.4,0.4)
      if fut is not None and i >= enc_seq_length:
        plt.subplot(num_rows, self.seq_length_, i + 1+self.seq_length_*2)
        if self.is_color_:
          plt.imshow(fut[i - enc_seq_length])
        else:
          plt.imshow(-data[i, :, :]+fut[i - enc_seq_length, :, :], cmap=plt.cm.jet,  interpolation="nearest")
          plt.clim(-0.4,0.4)
          error_sum += np.sum(np.abs(-data[i, :, :]+fut[i - enc_seq_length, :, :]))
      plt.axis('off')
    plt.draw()
    print output_file3
    plt.savefig(output_file3, bbox_inches='tight')
    return error_sum


