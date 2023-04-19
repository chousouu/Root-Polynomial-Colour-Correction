import numpy as np
import pandas as pd

def Responses_Reflectances(
        csv_resp = "/home/yasin/iitp/interview/colorchecker/Canon 600D.csv", 
        csv_refl = "/home/yasin/iitp/interview/colorchecker/CIE 1964.csv"):
    """returns tuple : (responses, reflectances)"""
    CIE1964_df = pd.read_csv(csv_refl)
    Canon_df   = pd.read_csv(csv_resp)

    return (Canon_df.to_numpy(), CIE1964_df.to_numpy())

def D65(
        csv_d65 = "/home/yasin/iitp/interview/colorchecker/d65_illum.csv"):

    return pd.read_csv(csv_d65).to_numpy()

def CheckerSpectrum( #380-730 wavelength
        csv_spectrum = "/home/yasin/iitp/interview/colorchecker/checkerlength.csv"):
    """returns tuple: (checker, wavelength)"""
    checker = pd.read_csv(csv_spectrum, index_col='Color name').to_numpy()
    # reshape to (y = 6 by x = 4) by bands, where 1 pixel is one color 
    checker = np.reshape(checker, (6, 4, checker.shape[1]))

    return (checker, np.arange(380, 730 + 1, 10))

