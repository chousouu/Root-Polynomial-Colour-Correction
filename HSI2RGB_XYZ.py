import h5py
import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import PchipInterpolator
from bisect import bisect
import pandas as pd
import GetData
import matplotlib.pyplot as plt

def HSI2XYZ(wY,HSI,ydim,xdim):
    """
    converts HSI to XYZ using CIE1964 color matching function under D65 illuminant

    wY: wavelengths in nm of hyperspectral data
    Y : HSI as a (#pixels x #bands) matrix,    
    dims: x & y size of image
    """
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
        
    # extrapolate to image wavelengths
    I = PchipInterpolator(wI,I,extrapolate=True)(wY)
    x = PchipInterpolator(w,x,extrapolate=True)(wY) 
    y = PchipInterpolator(w,y,extrapolate=True)(wY) 
    z = PchipInterpolator(w,z,extrapolate=True)(wY) 

    # cut at 780nm
    cut = bisect(wY, 780)
    HSI = HSI[:,0:cut]
    wY = wY[:cut]
    I = I[:cut]
    x = x[:cut]
    y = y[:cut]
    z = z[:cut]
    
    k = 1/np.trapz(I * y, wY)

    X = k * np.trapz(HSI @ np.diag(I * x), wY, axis=1)
    Y = k * np.trapz(HSI @ np.diag(I * y), wY, axis=1)
    Z = k * np.trapz(HSI @ np.diag(I * z), wY, axis=1)

    XYZ = np.array([X, Y, Z])
    
    X = np.reshape(XYZ[0,:],[ydim,xdim])
    Y = np.reshape(XYZ[1,:],[ydim,xdim])
    Z = np.reshape(XYZ[2,:],[ydim,xdim])
    
    return np.clip(np.transpose(np.array([X,Y,Z]),[1,2,0]), 0, 1)

def HSI2RGB(wY,HSI,ydim,xdim):
    """
    converts HSI to RGB using sensor sensitivities
    
    wY: wavelengths in nm of hyperspectral data  
    Y : HSI as a (#pixels x #bands) matrix,      
    dims: x & y size of image               
    """

    #wY : 400-730 wavelength

    # get camera responses
    D = GetData.responses_reflectances()[0]

    w = D[:,0] # len = 33, from 400 to 720
    r = D[:,1]
    g = D[:,2]
    b = D[:,3]
    
    #extrapolate to get color responses func   
    r = PchipInterpolator(w,r,extrapolate=True)(wY)  
    g = PchipInterpolator(w,g,extrapolate=True)(wY)  
    b = PchipInterpolator(w,b,extrapolate=True)(wY)  


    #  tried to fill with zeros
    #=====================================================
    # difference_bottom = int(abs((wY[0] - w[0]) / 10))
    # difference_top = int(abs((wY[-1] - w[-1]) / 10))

    # r = np.append(np.zeros(difference_bottom), r)
    # r = np.append(r, np.zeros(difference_top))
    # g = np.append(np.zeros(difference_bottom), g)
    # g = np.append(g, np.zeros(difference_top))
    # b = np.append(np.zeros(difference_bottom), b)
    # b = np.append(b, np.zeros(difference_top))
    #=====================================================
    
    above_cut = bisect(wY, wY[-1]) 

    HSI = HSI[:,0:above_cut]
    wY = wY[0:above_cut]
    r = r[0:above_cut]
    g = g[0:above_cut]
    b = b[0:above_cut]

    # Compute R,G,B for image

    k_r = 1 / integrate.trapezoid(r, wY)
    k_g = 1 / integrate.trapezoid(g, wY)
    k_b = 1 / integrate.trapezoid(b, wY)

    R = integrate.trapezoid(HSI * r, wY) * k_r
    G = integrate.trapezoid(HSI * g, wY) * k_g
    B = integrate.trapezoid(HSI * b, wY) * k_b

    R = np.reshape(R,[ydim,xdim])
    G = np.reshape(G,[ydim,xdim])
    B = np.reshape(B,[ydim,xdim])

    return np.transpose(np.array([R,G,B]),[1,2,0])
