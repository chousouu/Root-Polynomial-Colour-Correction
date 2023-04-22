import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import GetData
import HSI2RGB_XYZ as HSI

#TODO: rewrite (inefficient)
# WARNING : returns duplicates so before using apply filtered_P_combs()
def P_combs(sum, length = 3, combs = np.array([[0, 0, 0]]), 
                comb = np.array([])):
    for sum_i in range(0, sum + 1):
        if length > 2:
            for i in range(sum_i + 1):
                comb = np.append(comb, i)
                combs = P_combs(sum_i - i, combs = combs,
                                comb = comb, length = length - 1)
                comb = comb[:-1]
        else:
            for i in range(sum_i + 1):
                comb = np.append(comb, i)
                comb = np.append(comb, sum_i - i)
                combs = np.append(combs, [comb], axis=0)
                comb = comb[:-2]

    return combs    
#TODO: remove(part of temp solution)
def filtered_P_combs(sum, length = 3, combs = np.array([[0, 0, 0]]), 
                        comb = np.array([])):
    
    
    combs = P_combs(sum, length = length, comb = comb, combs=combs)
    return (np.unique(combs, axis = 0))[1:]

def RP_combs(degree):
    """returns array (3 , terms of Degree)"""
    RP_v = filtered_P_combs(degree)

    expanded = np.zeros((1, 3)) # 3 cuz represents R^x, G^y, B^z

    for row in RP_v:
        expanded = np.append(expanded, [row / np.sum(row)], axis = 0)
    
    return np.unique(expanded, axis = 0)[1:]

# TODO: rewrite?
def get_RP_vector(responses, degree, bias = False):
    #responses : 3xN responses (meaning 3xN RGB responses)
    """returns RGB vector extended to MxN, where M=number of terms and N responses""" 
    responses = preprocess_img(responses)

    R = responses[0, :]
    G = responses[1, :]
    B = responses[2, :]

    combs = RP_combs(degree) # 22x3, M = 22 , N = 3 ([r, g, b])

#!!!!TODO: change to M = **__func that counts amount like the one with PCC__**
    
    M = combs.shape[0] # amount of terms
    N = R.shape[0] #amount of responses
    
    expanded = np.zeros(shape=(1, N)) # needs to be NxM

    for i in range(M):
        elem = np.ones(shape=(1, N))
        if(combs[i][0] !=  0):
            elem *= R ** combs[i][0]
        if(combs[i][1] !=  0):
            elem *= G ** combs[i][1]
        if(combs[i][2] !=  0):
            elem *= B ** combs[i][2]
        expanded = np.append(expanded, elem, axis=0)

    # Not Sure
    # if bias == True : 
    #     expanded = np.append(expanded, np.ones(expanded.shape[1]))
    
    return expanded[1:, ]

# TODO unused maybe
def calc_tfactor():
    pass
    # return t_factor

def get_CCM(Q, R, degree, t_factor = 0):
    """
    returns ColourCorrection Matrix : 3xM
    Q : colorchecker reflectances
    R : colorchecker responses 
    """

    # if needed, reshape to 3xN     
    Q = preprocess_img(Q) 
    R = preprocess_img(R) 

    R = get_RP_vector(R, degree)

    RxRT = np.dot(R, R.T) #  MxN * NxM = MxM
    QxRT = np.dot(Q, R.T) #  3XN * NxM = 3xM

    ccm = np.dot(QxRT, np.linalg.inv(RxRT + t_factor*np.eye(RxRT.shape[0])))  
    return ccm

def apply_CCM(ccm, camera_responses, degree : int, scale = 1):
    """
    applies ccm to set of RGB responses
    camera_responses : Nx3 array containing r,g,b responses  
    scale : camera responses * scale first, then __expanded__ and multiplied 
    ccm : matrix 3xM, M = terms counter of chosen degree 
    """

    camera_responses = preprocess_img(camera_responses)  

    camera_responses = np.clip(camera_responses * scale, 0, 1)

    rho = get_RP_vector(camera_responses, degree)

    result_xyz = np.dot(ccm, rho)

    return result_xyz

def preprocess_img(image):
    if (np.ndim(image) == 3):
        image = np.transpose(np.array(image),[2,0,1])

        image = np.reshape(image, (image.shape[0], -1))
    elif ((np.ndim(image) == 2)):
        if((image.shape[0] != 3) and (image.shape[1] == 3)):
            image = image.T
    return image

def show_results(XYZ_REAL, XYZ_PRED, RGB):
    plt.subplots(2, 2, figsize=(20, 20))

    plt.subplot(2, 2, 1)
    plt.title("XYZ_REAL")
    plt.imshow(XYZ_REAL)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(2, 2, 2)
    plt.title("XYZ_PRED")
    plt.imshow(XYZ_PRED)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(2, 2, 3)
    plt.title("RGB")
    plt.imshow(RGB)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.show()



