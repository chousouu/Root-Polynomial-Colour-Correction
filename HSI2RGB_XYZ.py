import h5py
import numpy as np
import scipy.io as spio
from scipy.interpolate import PchipInterpolator
from bisect import bisect
import pandas as pd
import GetData

def HSI2XYZ(wY,HSI,ydim,xdim):
    """converts HSI to XYZ using CIE1964 color matching function under D65 illuminant"""
#|--------------------------------------------|
#| wY: wavelengths in nm of hyperspectral data|
#| Y : HSI as a (#pixels x #bands) matrix,    |
#| dims: x & y size of image                  |
#|--------------------------------------------|   
    # get reflectances
    D = GetData.responses_reflectances()[1]
    w = D[:,0]
    x = D[:,1]
    y = D[:,2]
    z = D[:,3]

    #illuminant d65 
    D  = GetData.D65()
    wI = D[:,0] # wavelength of D65 
    I  = D[:, 1] # values for each wavelength
    
    wY = np.array(wY)
    # extrapolate to image wavelengths
    I = PchipInterpolator(wI,I,extrapolate=True)(wY)
    x = PchipInterpolator(w,x,extrapolate=True)(wY) 
    y = PchipInterpolator(w,y,extrapolate=True)(wY) 
    z = PchipInterpolator(w,z,extrapolate=True)(wY) 

    # cut at 780nm
    cut = bisect(wY, 780)
    HSI = HSI[:,0:cut]/HSI.max()
    wY  = wY[:cut]
    I   = I[:cut]
    x   = x[:cut]
    y   = y[:cut]
    z   = z[:cut]
    
    k = 1/np.trapz(y * I, wY)

    X = k * np.trapz(HSI @ np.diag(I * x), wY, axis=1)
    Y = k * np.trapz(HSI @ np.diag(I * y), wY, axis=1)
    Z = k * np.trapz(HSI @ np.diag(I * z), wY, axis=1)

    XYZ = np.array([X, Y, Z])

    X = np.reshape(XYZ[0,:],[ydim,xdim])
    Y = np.reshape(XYZ[1,:],[ydim,xdim])
    Z = np.reshape(XYZ[2,:],[ydim,xdim])
    
    return np.clip(np.transpose(np.array([X,Y,Z]),[1,2,0]), 0, 1)

def HSI2RGB(wY,HSI,ydim,xdim):
    """converts HSI to RGB using sensor sensitivities"""
#|----------------------------------------------|
#| wY: wavelengths in nm of hyperspectral data  |
#| Y : HSI as a (#pixels x #bands) matrix,      |
#| dims: x & y size of image               |
#|----------------------------------------------|

    # get camera responses
    D = GetData.responses_reflectances()[0]

    w = D[:,0] # 33 from 400 to 720
    r = D[:,1]
    g = D[:,2]
    b = D[:,3]

    wY = np.array(wY)

    #extrapolate to get color responses func    
    r = PchipInterpolator(w,r,extrapolate=True)(wY)  
    g = PchipInterpolator(w,g,extrapolate=True)(wY)  
    b = PchipInterpolator(w,b,extrapolate=True)(wY)  

    # cut at 780nm
    cut = bisect(wY, 780)
    HSI = HSI[:,0:cut]/HSI.max()
    wY  = wY[:cut]
    r   = r[:cut]
    g   = g[:cut]
    b   = b[:cut]

    # Compute R,G & B for image
    R = np.trapz(HSI @ np.diag(r), wY, axis=1) / np.trapz(r, wY)
    G = np.trapz(HSI @ np.diag(g), wY, axis=1) / np.trapz(g, wY)
    B = np.trapz(HSI @ np.diag(b), wY, axis=1) / np.trapz(b, wY)

    print(R.shape)

    RGB = np.array([R, G, B])
    R = np.reshape(RGB[0,:],[ydim,xdim])
    G = np.reshape(RGB[1,:],[ydim,xdim])
    B = np.reshape(RGB[2,:],[ydim,xdim])

    return np.clip(np.transpose(np.array([R,G,B]),[1,2,0]), 0 , 1)

