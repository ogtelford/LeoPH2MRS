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

# define constants, information about the star
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


# load the data -- here, we are only searching Channel 3 at the MRS pointing containing the H II region,
# but the same code can be re-run for all 4 pointings (data available upon request)
filesdir = 'data/'
ch = 3
f = fits.open(filesdir + 'destripe_v0_ch{}_s3d.fits'.format(ch))
fluxes_init = f['SCI'].data
header = f['SCI'].header
w = wcs.WCS(header)
# set up array of cube wavelengths
wave_orig = np.arange(header['NAXIS3']) * header['CDELT3'] + header['CRVAL3'] 
# rest-frame wavelengths, corrected for Leo P's systemic velocity 
wave = wave_orig / (1 + redshift)
# convert fluxes to units of erg/cm^2/s/micron/Sr
fluxes = fluxes_init * 1e-17 * 3e10 / ((wave[:, np.newaxis, np.newaxis] / 1e4) ** 2) / 1e4
# integrate surface brightness over wavelengths of the [Ne III] nebular line to make a nebular emission map
wh = (wave_orig > 15.55) * (wave_orig < 15.58)
intlight = np.trapz(fluxes[wh], wave[wh], axis=0)

# plot the nebular emission map to orient us to the location of the H II region
fig = plt.figure(figsize=(8, 4.8))
ax = fig.add_subplot(111, projection=w, slices=('x', 'y', 1000))
im = ax.imshow(intlight, vmin=0, cmap='magma', origin='lower')
ax.set_xlabel('Right Ascension')
ax.set_ylabel('Declination', labelpad=-1)
ax.coords[0].set_ticks(spacing=7.5 * u.arcsec)
ax.coords[1].set_ticks(spacing=4 * u.arcsec)
cb = plt.colorbar(im, ax=ax)
cb.set_label(r'[Ne III] 15.5 $\mu$m SB (erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$)', fontsize=14)
fig.tight_layout()
fig.subplots_adjust(left=-0.2)

# now, iterate over all spaxels in the cube and fit a Gaussian at the location of the H2 S(1) line
# change plotfit to True to generate plots of all Gaussian fits where S(1) emission is detected
plotfit = True
xs_detected = []
ys_detected = []
for xx in np.arange(intlight.shape[1]):
    for yy in np.arange(intlight.shape[0]):
        if not np.isnan(intlight[yy, xx]):
            # get the spectrum in 1 pixel at a time
            spec_ext = fluxes[:, yy, xx]
            # define the rest wavelength of the H2 S(1) line  
            line_center = 17.0348
            # only consider wavelengths within +/- 700 km/s of expected line center
            cont_lims = [line_center * (1 - 700 / 3e5), line_center * (1 + 700 / 3e5)]
            gauss_lims = [line_center * (1 - 240 / 3e5), line_center * (1 + 240 / 3e5)]
            whcont = ((wave > cont_lims[0]) * (wave < gauss_lims[0])) + ((wave > gauss_lims[1]) * (wave < cont_lims[1]))
            cont_rms = np.nanstd(spec_ext[whcont])
            whfit = (wave > cont_lims[0]) * (wave < cont_lims[1])
            # calculate the MRS spectral resolution (sigma) at 17 microns following Argyriou+ (2023)
            resol = 4603 - (128 * line_center)
            sigma_init_kms = 3e5 / resol / 2.355  # last value is conversion between FWHM and sigma
            # set up initial guesses and limits; require velocities close to systemic and sigmas within 1% of instrumental resolution
            init = [1e-6, 260., sigma_init_kms, 1e-5]
            low = [-np.inf, 220., sigma_init_kms * 0.99, 0.]
            up = [np.inf, 300., sigma_init_kms * 1.01, np.inf]
            line_centers = [line_center]
            # now fit a Gaussian to the H2 line -- use observed (not rest) wavelengths since we require a velocity close to systemic
            # try/except statement deals with cases where the fitting fails because there is no emission line to fit
            try:
                # use the continuum rms as surface brightness uncertainty estimate in the fitting
                err_ext = np.ones(whfit.sum()) * cont_rms
                fit_gauss = least_squares(residuals, init, bounds=(low, up),
                                          args=(wave_orig[whfit], spec_ext[whfit],
                                                err_ext, line_centers))
                # calculate the surface brightness of the line -- need to convert sigma_v to sigma_wave
                # result is in erg/cm^2/s/Sr
                s1_sb = np.sqrt(2 * np.pi) * fit_gauss.x[2] / 3e5 * line_centers[0] * fit_gauss.x[3]
                # calculate uncertainty from fit statistics
                J = fit_gauss.jac
                cov = np.linalg.inv(J.T.dot(J))
                stdev = np.sqrt(np.diagonal(cov))
                frac_errs = stdev / fit_gauss.x
                s1_sb_err = s1_sb * np.sqrt((frac_errs[2]) ** 2 + (frac_errs[3]) ** 2)
            except (ValueError, LinAlgError):
                continue

            # require a 3-sigma detection to consider H2 S(1) detected in this spaxel
            if (s1_sb > 3 * s1_sb_err):  
                xs_detected.append(xx)
                ys_detected.append(yy)
                # print information about the detected H2 S(1) lines
                print('({:.0f},{:.0f})'.format(xx, yy))
                print('H2 S(1) line flux {:.1e}; {:.1f} sigma'.format(s1_sb, (s1_sb / s1_sb_err)))
                print('Velocity: {:.0f} km/s'.format(fit_gauss.x[1]))
                # plot the location of this detection on the [Ne III] emission map
                ax.scatter(xx, yy, color='C0', s=5)
                if plotfit:
                    # make a new figure and plot the spectrum around the S(1) line with the best-fit Gaussian overplotted
                    specfig, specax = plt.subplots(figsize=(6, 4.8))
                    specax.axvline(line_center, color='k', zorder=1, linewidth=1)
                    specax.text(line_center + 0.03, 0.0003, r'H$_2$ S(1)', fontsize=16)
                    specax.text(line_center + 0.03, 0.0003, '\n%.1fx10$^{-7}$ \nerg/s/cm$^2$/Sr' % (s1_sb / 1e-7),
                                fontsize=14, va='top')
                    specax.plot(wave, spec_ext, label='({},{})'.format(xx, yy))
                    specax.axhline(0, color='C7', linewidth=1, zorder=0)
                    specax.set_xlabel(r'Wavelength ($\mu$m)')
                    specax.set_ylabel(r'SB (erg cm$^{-2}$ s$^{-1}$ $\mu$m$^{-1}$ Sr$^{-1}$)')
                    # make a version of the array to plot that only includes wavelengths used in the fitting
                    spec_ext_fit = np.copy(wave)
                    spec_ext_fit[~whfit] = np.nan
                    specax.plot(spec_ext_fit, spec_ext, color='C4', label='Used in Fit')
                    model = gaussian_fit_bkg(fit_gauss.x, wave_orig[whfit], line_centers)
                    specax.plot(wave[whfit], model, color='C1', label='Gaussian Fit')
                    specax.set_title('({},{}): '.format(xx, yy) + 'Detected')
                    specax.set_ylim(-0.0001, 0.0005)
                    specax.set_xlim(16.35, 17.75)
                    specfig.tight_layout()
            else:
                # not a significant detection, move on
                pass