import numpy as np
import pandas as pd

DATA_PATH = "/home/yasin/iitp/interview/Root-Polynomial-Colour-Correction/data/"

def responses_reflectances(
        csv_resp = DATA_PATH + "Canon 600D.csv", 
        csv_refl = DATA_PATH + "CIE 1964.csv"):
    """returns tuple : (responses, reflectances)"""

    CIE1964_df = pd.read_csv(csv_refl)
    Canon_df = pd.read_csv(csv_resp)
    CIE1964_df = CIE1964_df[["wavelength", "x_bar", "y_bar", "z_bar"]]
    Canon_df = Canon_df[["wavelength", "r", "g", "b"]] 

    return (Canon_df.to_numpy(), CIE1964_df.to_numpy())

def D65(
        csv_d65 = DATA_PATH + "d65_illum.csv"):
    """returns D65 illuminant data"""

    return pd.read_csv(csv_d65).to_numpy()

def checker_spectrum(
        csv_spectrum = DATA_PATH + "checkerlength.csv"):
    """returns tuple: (checker, wavelength) [380-730 wavelength]"""
    
    checker = pd.read_csv(csv_spectrum, index_col='Color name').to_numpy()
    # reshape to (y = 6 by x = 4) by bands, where 1 pixel is one color 
    checker = np.reshape(checker, (6, 4, checker.shape[1]))

    return (checker, np.arange(380, 730 + 1, 10))

