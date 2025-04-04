"""
Script to perform calculations shown in Figure 2 of Telford et al. (2025)
"""
from scipy.optimize import least_squares
from astropy.io import fits
from astropy import wcs
from astropy import units as u
import numpy as np
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

# define constants, info about the star
lp26_ra = 155.438
lp26_dec = 18.088
distance = 1.62  # Mpc
redshift = 260. / 3e5  # approximate, based on earlier data

# set up dicts of lines in each channel
waves_dict = {2: [8.9914, 9.6649, 10.5105],  
              3: [12.2786, 12.37, 12.8135, 15.5551, 17.0348], 
              4: [18.713, 21.8302, 28.2188]}
names_dict = {2: ['[Ar III]', r'H$_2$ S(3)', '[S IV]'],
              3: [r'H$_2$ S(2)', r'Hu $\alpha$', '[Ne II]', '[Ne III]', r'H$_2$ S(1)'],
              4: ['[S III]', '[Ar III]', r'H$_2$ S(0)']}
intlimits_dict = {2: [10.51, 10.54], 3: [15.55, 15.58], 4: [18.7, 18.73]}
cblabels_dict = {2: r'[S IV] 10.5$\mu$m', 3: r'[Ne III] 15.5$\mu$m', 4: r'[S III] 18.7$\mu$m'}

# define line centers and wavelength limits for Gaussian fitting
line_centers_dict = {r'H$_2$ S(3)': 9.6649, r'H$_2$ S(2)': 12.2786, r'H$_2$ S(1)': 17.0348}
cont_lims_dict = {r'H$_2$ S(3)': [9.64, 9.69], r'H$_2$ S(2)': [12.25, 12.31],
                  r'H$_2$ S(1)': [17.00, 17.07]}  # +/- 700 km/s
gauss_lims_dict = {r'H$_2$ S(3)': [9.65, 9.68], r'H$_2$ S(2)': [12.26, 12.29],
                   r'H$_2$ S(1)': [17.02, 17.05]}  # +/- 300 km/s

# set up dictionaries to store measured H2 line intensities and uncertainties
intensity = dict()
intensity_err = dict()
# translate between naming conventions in different sections of code
keys_dict = {'H$_2$ S(1)': 'H200S1', 'H$_2$ S(2)': 'H200S2', 'H$_2$ S(3)': 'H200S3'}
filesdir = 'data/'

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


for ch in [2, 3]:
    pass
