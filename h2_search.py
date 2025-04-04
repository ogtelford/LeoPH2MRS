"""
Script to search for H2 S(1) emission in Leo P as described in Telford et al. (2025)
"""

from numpy.linalg import LinAlgError
from scipy.optimize import least_squares
from scipy import optimize as opt
from scipy import interpolate
from astropy.io import fits
from astropy import wcs
from astropy import units as u
import numpy as np
import warnings
from astropy.utils.exceptions import AstropyWarning
from astropy.coordinates import SkyCoord
warnings.simplefilter('ignore', category=AstropyWarning)
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
plt.ion()

# define constants, info about the star
lp26_ra = 155.438
lp26_dec = 18.088
distance = 1.62  # Mpc
redshift = 260. / 3e5  # approximate, based on earlier data


# function definitions


def gaussian_fit_bkg(p, x, linewaves):
    # fit for the velocity of the spectrum w/ simultaneous Gaussian fits to Balmer lines
    # p[0] = continuum level; p[1] = velocity; p[2] = sigma; p[3:] = amplitude of each line
    modelspec = np.zeros(len(x)) + p[0]
    for ii in np.arange(len(linewaves)):
        v = (x - linewaves[ii]) / linewaves[ii] * 3e5
        modelspec += np.exp(-(v - p[1]) ** 2 / (2 * p[2] ** 2)) * p[ii + 3]
    return modelspec


def residuals(p, x, data, unc, linewaves):
    return (data - gaussian_fit_bkg(p, x, linewaves)) / unc


# load the data
filesdir = 'data/'
ch = 3
f = fits.open(filesdir + 'destripe_v0_ch{}_s3d.fits'.format(ch))
fluxes_init = f['SCI'].data
header = f['SCI'].header
w = wcs.WCS(header)
wave_orig = np.arange(header['NAXIS3']) * header['CDELT3'] + header['CRVAL3']  # set up array of cube wavelengths
wave = wave_orig / (1 + redshift)
# convert fluxes to units of erg/cm^2/s/micron/Sr
fluxes = fluxes_init * 1e-17 * 3e10 / ((wave[:, np.newaxis, np.newaxis] / 1e4) ** 2) / 1e4