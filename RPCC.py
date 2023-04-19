import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import GetData
import HSI2RGB_XYZ as HSI

#TODO: rewrite  (no leave in recursion)
def P_combs(sum, length = 3, combs = np.array([[0, 0, 0]]), comb = np.array([])):
    for sum_i in range(0, sum + 1):
        if length > 2:
            for i in range(sum_i + 1):
                comb = np.append(comb, i)
                combs = P_combs(sum_i - i, combs = combs,comb = comb, length = length - 1)
                comb = comb[:-1]
        else:
            for i in range(sum_i + 1):
                comb = np.append(comb, i)
                comb = np.append(comb, sum_i - i)
                combs = np.append(combs, [comb], axis=0)
                comb = comb[:-2]

    return combs    
#TODO: remove(part of temp solution)
def filtered_P_combs(sum, length = 3, combs = np.array([[0, 0, 0]]), comb = np.array([])):
    combs = P_combs(sum, length = length, comb = comb, combs=combs)
    return (np.unique(combs, axis = 0))[1:]

# TODO: rewrite?
#extended by adding additional rpcc terms
def getRPVector(responses, degree, bias = False):
    #responses  : N responses (meaning Nx3 RGB responses)
    # returns NxM vector, where M - amount of terms of polynom of degree == degree 
    R = responses[:, 0]
    G = responses[:, 1]
    B = responses[:, 2]

    print(f"B = {B.shape}, R = {R.shape}")
#!!!!TODO: change to M = **__func that counts amount like the one with PCC__**
    M = (RP_combs(degree).shape)[0]
    expanded = np.zeros((1, M)) # needs to be NxM
    combs = RP_combs(degree)
    rows = R.shape[0]
    expanded = np.zeros(shape=(1, R.shape[0]))
    for term in combs: # term e.g rg  == [1, 1, 0]
        elem = np.ones(shape = (1, R.shape[0]))
#TODO: govnocode
        if(term[0] != 0):
            elem *= R**term[0]
        if(term[1] != 0):
            elem *= G**term[1]
        if(term[2] != 0):
            elem *= B**term[2]
        expanded = np.append(expanded, elem, axis = 0)

    # Not Sure
    # if bias == True : 
    #     expanded = np.append(expanded, np.ones(expanded.shape[1]))

    return expanded[1:].T

def RP_combs(degree):
    RP_v = filtered_P_combs(degree)

    expanded = np.zeros((1, 3)) # 3 cuz represents R^x, G^y, B^z

    for row in RP_v:
        expanded = np.append(expanded, [row / np.sum(row)], axis = 0)
    
    return np.unique(expanded, axis = 0)[1:]

def CalcTFactor():
    pass
    # return t_factor

def GetCCM(Q, R, t_factor = 0):
    # Q : reflectances Nx3
    # R : camera resp  Nx3

    # if (t_factor != 0):
    #     t_factor = CalcTFactor()

    RxRT = np.dot(R.T, R)  # 3XN * NX3
    R_TxQ = np.dot(R.T, Q) # 3XN * NX3

    print(f"det = {np.linalg.det(RxRT + t_factor*np.eye(RxRT.shape[0]))}")

    ccm = np.dot(np.linalg.inv(RxRT + t_factor*np.eye(RxRT.shape[0])), R_TxQ)  
    #3XN * NX3 * 3XN*NX3 = 3X3
    return ccm

def ApplyCCM(image, camera_responses, reflectances, degree, scale = 1, t_factor = 2.3):
    # camera_responses : Nx3 array containing r,g,b responses (in range 0-1) of img 
    # ^ need to change cuz picture is NxMx3
    # reflectances : Nx3 array containing x,y,z (in range 0-1)
    # scale : camera responses * scale first, then __expanded__ and multiplied 

    camera_responses *= scale
    camera_responses = np.clip(camera_responses, 0, 1)

    #preprocess data

    rgb_x = getRPVector(image, degree) #image expanded

    R = getRPVector(camera_responses, degree) # camera responses expanded to root polynom of degree 
                                              # needed for ccm
    ccm = GetCCM(Q = reflectances, R = R, t_factor = t_factor)

    print(rgb_x.shape, ccm.shape)
    predicted_responses = rgb_x.dot(ccm)

    return predicted_responses #predicted?

def PreprocessImg(image):
    if (np.ndim(image) == 3):
        image = np.reshape(image, (image.shape[0] * image.shape[1], 3))
    if (len(image[image > 1])):
        image = image / 255 #normalize input
    return image

# HSI_PATH = "/home/yasin/iitp/interview/li_ds/"
# HSI_NAME = "2019-08-26_006.h5" 


def main():

    HSI_data = GetData.CheckerSpectrum()[0]
    (ydim, xdim, zdim) = HSI_data.shape
    # Load wavelengths of hyperspectral data
    wl = GetData.CheckerSpectrum()[1]

        #TODO: fix so works on random pics
    # h5file = h5py.File("/home/yasin/iitp/interview/li_ds/2019-09-18_003.h5", 'r')
    # HSI_data = h5file['img\\']
    # HSI_data = np.reshape(HSI_data, [HSI_data.shape[1], HSI_data.shape[2], HSI_data.shape[0]])
    # print((HSI_data.shape))
    # (ydim, xdim, zdim) = HSI_data.shape

    # Reorder data so that each column holds the spectra of of one pixel
    HSI_data = np.reshape(HSI_data, [-1,zdim])/HSI_data.max()

    XYZ_image = HSI.HSI2XYZ(wl, HSI_data, xdim, ydim)

    RGB_IMAGE = HSI.HSI2RGB(wl, HSI_data, xdim, ydim)


    plt.subplots(1 , 2, figsize=(20, 20))
    plt.subplot(1, 2, 1)
    plt.title("RGB")
    plt.imshow(RGB_IMAGE)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.title("XYZ")
    plt.imshow(XYZ_image)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.show()



if __name__ == '__main__':
    # cpm.MacBeth_ColorChecker_patch_plot()
    main()
