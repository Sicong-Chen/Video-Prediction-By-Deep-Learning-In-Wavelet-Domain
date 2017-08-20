import numpy as np
import pywt
#import cv2
from matplotlib import pyplot as plt
import constants as c


def threeDWT(clips):


    index = len(clips.shape)


    coeffs1 =pywt.wavedecn(clips, wavelet='bior1.1', level=1,axes = [index - 3 ,index - 2])
    arr1, coeff_slices1 = pywt.coeffs_to_array(coeffs1, axes =[index - 3 ,index - 2])
    #do bior1.1 WT at axes0,1 to get arr1
    coeffs2= pywt.wavedecn(arr1, wavelet='haar', level=1,axes = [index - 1])
    arr2, coeff_slices2 = pywt.coeffs_to_array(coeffs2, axes =[index - 1])
    #do haar WT at axes2(time domain) to get arr2
    # Height,Length,Width = arr2.shape
    # List =[arr2[0:Height/2,0:Length/2,0:Width/2],
           # arr2[0:Height/2,Length/2:0,0:Width/2],
           # arr2[Height/2:0,0:Length/2,0:Width/2],
           # arr2[Height/2:0,Length/2:0,0:Width/2],
           # arr2[0:Height/2,0:Length/2,Width/2:0],
           # arr2[0:Height/2,Length/2:0,Width/2:0],
           # arr2[Height/2:0,0:Length/2,Width/2:0],
           # arr2[Height/2:0,Length/2:0,Width/2:0]]
          #Length 8 List,each is a 3D array
    return arr2  #return big cube after WT



def threeIDWT(Preds):

    index = len(Preds.shape)

    """
    coeffs1 =pywt.wavedecn(np.zeros_like(Preds), wavelet='bior1.1', level=1,axes = [0,1])
    arr1, coeff_slices1 = pywt.coeffs_to_array(coeffs1, axes =[0,1])
    coeffs2= pywt.wavedecn(arr1, wavelet='haar', level=1,axes = [2])
    arr2, coeff_slices2 = pywt.coeffs_to_array(coeffs2, axes =[2])
    #to get coeff_slices1&coeff_slices2 to reconstruction
    """
    coeffs1 =pywt.wavedecn(np.zeros_like(Preds), wavelet='haar', level=1,axes = [index -1])
    arr1, coeff_slices1 = pywt.coeffs_to_array(coeffs1, axes =[index -1])
    coeffs2= pywt.wavedecn(arr1, wavelet='bior1.1', level=1,axes = [index - 3, index - 2])
    arr2, coeff_slices2 = pywt.coeffs_to_array(coeffs2, axes =[index - 3, index - 2])
    #to get coeff_slices1&coeff_slices2 to reconstruction


    coeffs_1 = pywt.array_to_coeffs(Preds,coeff_slices1,'wavedecn')
    arr_level1 = pywt.waverecn(coeffs_1, wavelet='bior1.1',axes =[index -1])
    #axes 2 reconstruction
    coeffs_2 = pywt.array_to_coeffs(arr_level1, coeff_slices2,'wavedecn')
    recon_frames = pywt.waverecn(coeffs_2, wavelet='haar',axes =[index - 3, index - 2])
    #axes0,1 reconstuction
    return recon_frames



def split(batch): #4D batch
    Height,Length,Width = batch.shape
    List = [batch[:,0:Height/2,0:Length/2,0:Width/2],
            batch[:,0:Height/2,Length/2:0,0:Width/2],
            batch[:,Height/2:0,0:Length/2,0:Width/2],
            batch[:,Height/2:0,Length/2:0,0:Width/2],
            batch[:,0:Height/2,0:Length/2,Width/2:0],
            batch[:,0:Height/2,Length/2:0,Width/2:0],
            batch[:,Height/2:0,0:Length/2,Width/2:0],
            batch[:,Height/2:0,Length/2:0,Width/2:0]]

    return List #return List[4D,4D,4D,4D,4D,4D,4D,4D]

def fuse(List,Len):
    batch = np.empty([c.BATCH_SIZE,
                      c.FULL_HEIGHT,
                      c.FULL_WIDTH,
                      (c.NUM_INPUT_CHANNEL * Len)])
    batch[:,0:Height/2,0:Length/2,0:Width/2] = List[0]
    batch[:,0:Height/2,Length/2:0,0:Width/2] = List[1]
    batch[:,Height/2:0,0:Length/2,0:Width/2] = List[2]
    batch[:,Height/2:0,Length/2:0,0:Width/2] = List[3]
    batch[:,0:Height/2,0:Length/2,Width/2:0] = List[4]
    batch[:,0:Height/2,Length/2:0,Width/2:0] = List[5]
    batch[:,Height/2:0,0:Length/2,Width/2:0] = List[6]
    batch[:,Height/2:0,Length/2:0,Width/2:0] = List[7]
    return batch
