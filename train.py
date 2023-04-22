import numpy as np
import matplotlib.pyplot as plt
import h5py

import GetData
import HSI2RGB_XYZ as HSI
import RPCC as RP

def get_best_CCM(degree):
    HSI_data = GetData.checker_spectrum()[0]
    (ydim, xdim, zdim) = HSI_data.shape
    # Load wavelengths of hyperspectral data
    wl = GetData.checker_spectrum()[1]

    HSI_data = np.reshape(HSI_data, [-1, zdim])/HSI_data.max()

    XYZ_true = HSI.HSI2XYZ(wl, HSI_data, xdim, ydim)

    RGB_true = HSI.HSI2RGB(wl, HSI_data, xdim, ydim)

    #reshape and transfer to 2dim for train
    X = np.reshape(np.transpose(RGB_true,[2,0,1]), (3, -1))
    Y = np.reshape(np.transpose(XYZ_true, [2,0,1]), (3, -1))

    return train(X, Y, degree)

def train(X, Y, degree):
    """
    returns best colour correction matrix using angle metric
    by running Leave-One-Out Cross Validation.
    """
    best_ccm = None
    best_score = 999 
    for i in range(X.shape[1]):
        X_train = np.append(X[:, :i], X[:, i+1:], axis=1)
        X_test = np.reshape(X[:, i], (3, 1))
        Y_train = np.append(Y[:, :i], Y[:, i+1:], axis=1)
        Y_test = np.reshape(Y[:, i], (3, 1))

        ccm = RP.get_CCM(Y_train, X_train, degree, t_factor = 0)

        Y_pred = RP.apply_CCM(ccm, X_test, degree)

        score = scoring(Y_pred, Y_test)
        if (score < best_score):
            best_score = score
            best_ccm = ccm

    return best_ccm

def scoring(Y_pred, Y_true):
    """returns angle between predicted and true vectors """
    mul_abs = np.sum(Y_pred**2) * np.sum(Y_true**2) 

    error = np.arccos(np.sum(Y_true * Y_pred) / (mul_abs**0.5))

    return error


def main():
    DEGREE = 4
    
    HSI_name = "2019-09-08_015.h5"
    # HSI_name = "2019-08-28_016.h5" # green(red) leaves
    # HSI_name = "2019-09-18_003.h5" # colorchecker image
    h5file = h5py.File("/home/yasin/iitp/interview/li_ds/" + HSI_name, 'r')
    HSI_data = h5file['img\\']
    
    # (bands, pixels x pixels) -> (pixels x pixels, bands)
    HSI_data = np.transpose(HSI_data, (1, 2, 0))
    
    #remember sizes
    (ydim, xdim, zdim) = HSI_data.shape
    wl = np.arange(400, 730 + 1, 10) #wavelength of hs images

    # Reorder data so that each column holds the spectra of of one pixel
    HSI_data = np.reshape(HSI_data, [-1,zdim])

    XYZ_true = HSI.HSI2XYZ(wl, HSI_data, xdim, ydim)

    RGB_true = HSI.HSI2RGB(wl, HSI_data, xdim, ydim)

    ccm = get_best_CCM(DEGREE)

    XYZ_pred = np.reshape((RP.apply_CCM(ccm, RGB_true, DEGREE)).T, (xdim, ydim, 3))
   
    RP.show_results(XYZ_REAL=XYZ_true, XYZ_PRED=XYZ_pred, RGB=RGB_true)



if __name__ == '__main__':
    main()