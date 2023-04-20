from sklearn.model_selection import LeaveOneOut
import numpy as np
import GetData
import h5py
import HSI2RGB_XYZ as HSI
import RPCC as RP

def get_best_CCM(X, Y, degree):
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
        # print(f"Y_TEST = {Y_test.shape}, y_train = {Y_train.shape}, X_train = {X_train.shape} XTEST = {X_test.shape} ")

        ccm = RP.get_CCM(Y_train, X_train, degree, t_factor = 0)

        Y_pred = RP.apply_CCM(ccm, X_test, degree)

        score = scoring(Y_pred, Y_test)
        if (score < best_score):
            best_score = score
            best_ccm = ccm

    return best_ccm

def scoring(Y_pred, Y_true):
    """returns angle between predicted and true vectors """
    mul_abs = (Y_pred[0]**2 + Y_pred[1]**2 + Y_pred[2]**2) * \
              (Y_true[0]**2 + Y_true[1]**2 + Y_true[2]**2) 

    error = np.arccos(np.sum(Y_true * Y_pred) / (mul_abs**0.5))

    return error



def main():
    DEGREE = 4

#==========================COLORCHECKER=======================================
    HSI_data = GetData.checker_spectrum()[0]
    (ydim, xdim, zdim) = HSI_data.shape
    # Load wavelengths of hyperspectral data
    wl = GetData.checker_spectrum()[1]
#=============================================================================


#===========================================HSI====================================================
    # h5file = h5py.File("/home/yasin/iitp/interview/li_ds/2019-09-18_003.h5", 'r')
    # HSI_data = h5file['img\\']
    # HSI_data = np.reshape(HSI_data, [HSI_data.shape[1], HSI_data.shape[2], HSI_data.shape[0]])

    # # HSI_data = HSI_data[:64, :64, :]

    # (ydim, xdim, zdim) = HSI_data.shape
    # wl = np.arange(400, 730 + 1, 10)
#==================================================================================================

    # Reorder data so that each column holds the spectra of of one pixel
    HSI_data = np.reshape(HSI_data, [-1,zdim])/HSI_data.max()

    XYZ_true = HSI.HSI2XYZ(wl, HSI_data, xdim, ydim)

    RGB_true = HSI.HSI2RGB(wl, HSI_data, xdim, ydim)

    ccm = RP.get_CCM(XYZ_true, RGB_true, DEGREE, t_factor = 0)

    copy = ccm
    XYZ_pred = RP.apply_CCM(ccm, RGB_true, DEGREE)

    #transfer to 2dim
    X = np.reshape(RGB_true, (3, RGB_true.shape[0] * RGB_true.shape[1]))
    Y = np.reshape(XYZ_true, (3, XYZ_true.shape[0] * XYZ_true.shape[1]))

    best_ccm = get_best_CCM(X, Y, DEGREE)

    RESULT = RP.apply_CCM(best_ccm, RGB_true, DEGREE)

    # #show results
    # xyz_pred = np.reshape(xyz_pred,(xdim, ydim, 3))
    # XYZ_image = np.reshape(XYZ_image, (xdim, ydim, 3))
    # RGB_IMAGE = np.reshape(XYZ_image, (xdim, ydim, 3))
    RESULT = np.reshape(RESULT, (xdim, ydim, 3))
    RP.show_results(XYZ_true, RESULT)





if __name__ == '__main__':
    main()