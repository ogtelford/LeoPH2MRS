"""
Script to perform calculations shown in Figure 2 of Telford et al. (2025)
"""
from scipy.optimize import least_squares
from astropy.io import fits
from astropy import wcs
from astropy import units as u
import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)
plt.ion()  # use interactive plotting in iPython; can comment out if desired.

# define constants, info about the star
lp26_ra = 155.438
lp26_dec = 18.088
distance = 1.62  # Mpc
redshift = 260. / 3e5  # approximate, based on earlier data

# set up dicts of lines in each channel (H2 and some nebular lines; not all are used here)
waves_dict = {2: [8.9914, 9.6649, 10.5105],  
              3: [12.2786, 12.37, 12.8135, 15.5551, 17.0348]}
names_dict = {2: ['[Ar III]', r'H$_2$ S(3)', '[S IV]'],
              3: [r'H$_2$ S(2)', r'Hu $\alpha$', '[Ne II]', '[Ne III]', r'H$_2$ S(1)']}
intlimits_dict = {2: [10.51, 10.54], 3: [15.55, 15.58]}
cblabels_dict = {2: r'[S IV] 10.5$\mu$m', 3: r'[Ne III] 15.5$\mu$m'}

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

# function definitions (same as in h2_search.py)


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


# set up a figure to populate later -- similar to Figure 2 in Telford+ (2025)
width = 10
fig = plt.figure(figsize=(width, width * 0.4))
gs = GridSpec(2, 3, figure=fig, width_ratios=[5, 3, 5])
ax_s1 = fig.add_subplot(gs[:, 0])
ax_s3 = fig.add_subplot(gs[0, 1])
ax_s2 = fig.add_subplot(gs[1, 1])
ax_fit = fig.add_subplot(gs[:, 2])
ax_s1.set_xlabel(r'Rest wavelength ($\mu$m)')
ax_s2.set_xlabel(r'Rest wavelength ($\mu$m)')
ax_s1.set_ylabel(r'Surface brightness (10$^{-4}$ erg cm$^{-2}$ s$^{-1}$ $\mu$m$^{-1}$ sr$^{-1}$)')
ax_fit.set_xlabel(r'$E_u\,/\,k_\mathrm{B}$ (K)')
ax_fit.set_ylabel(r'$\ln(N_u\,/\,g_u\,/\,$cm$^{-2}$)')
fig.tight_layout(pad=0.25)
axes_dict = {r'H$_2$ S(3)': ax_s3, r'H$_2$ S(2)': ax_s2, r'H$_2$ S(1)': ax_s1}
fig.add_subplot(gs[:, 1], frame_on=False)
plt.tick_params(labelcolor='none', bottom=False, left=False)
plt.ylabel(r'Surface brightness (10$^{-4}$ erg cm$^{-2}$ s$^{-1}$ $\mu$m$^{-1}$ sr$^{-1}$)')

# loop over channels 2 and 3 to measure the S(3) and S(2), S(1) lines, respectively
for ch in [2, 3]:
    filesdir = 'data/'
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
    fluxes /= 1e-4  # just to get reasonable units on y-axes; will undo this scaling later!
    
    # make an emission line map image, even though we aren't plotting it, just to select the H2-detected region
    wh = (wave_orig > intlimits_dict[ch][0]) * (wave_orig < intlimits_dict[ch][1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        intlight = np.nanmean(fluxes[wh], axis=0)

    # get the pixel scale in this channel (spaxels are smaller in Ch 2 vs. Ch 3)
    arcsec_per_pixel = f['SCI'].header['PIXAR_A2'] ** 0.5
    # keep the diameter fixed across both channels and calculate how many pixels that corresponds to in this channel
    radius_pixels = 0.668 / 2 / arcsec_per_pixel  # radius = 0.5 FWHM at 17 micron
    # read the center of the H2 detected region from text file
    with open('coordinates_H2.txt', 'r') as coordfile:
        coordstring = coordfile.readline()
    center_ra = float(coordstring.split(', ')[0])
    center_dec = float(coordstring.split(', ')[1])
    # get the pixel location in this cube
    h2_center_coords = w.wcs_world2pix(center_ra, center_dec, 1, 1)

    # set up a mask for the pixels in the emission line map
    xs = np.arange(intlight.shape[0])  # vertical in a matplotlib figure
    ys = np.arange(intlight.shape[1])  # horizontal
    xx, yy = np.meshgrid(xs, ys)
    xct = xx - (h2_center_coords[1])  # find distance from center
    yct = yy - (h2_center_coords[0])
    mask_3d = np.zeros(fluxes.shape, dtype=bool)

    # now finally calculate the average 1D spectrum within the H2-detected region
    extract_region = np.sqrt(xct ** 2 + yct ** 2) < radius_pixels 
    mask_3d[:, extract_region.T] = 1
    spec_ext = fluxes.mean(axis=(1, 2), where=mask_3d) 

    # plot the extracted 1D spectrum -- similar to Figure 2 in Telford+ (2025)
    annotate_waves = waves_dict[ch]
    annotate_names = names_dict[ch]
    for ii in range(len(annotate_waves)):
        # loop over ONLY the H2 lines in the line list
        if annotate_names[ii][:2] == 'H$':
            # pick which axis we will use in the figure based on the line name
            ax = axes_dict[annotate_names[ii]]
            # plot the MRS spectrum around this line! 
            ax.plot(wave, spec_ext, label='MIRI-MRS data', linewidth=1, color='k', drawstyle='steps-mid')
            ax.axhline(0, color='C7', linewidth=1, zorder=0)
            ax.set_xlim(annotate_waves[ii] - 0.1, annotate_waves[ii] + 0.1)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
            ax.set_ylim(-2, 2)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.text(0.95, 0.95, annotate_names[ii], ha='right', va='top',
                    transform=ax.transAxes, fontsize=7)

            # fit a Gaussian to this H2 line
            line = annotate_names[ii]
            line_center = line_centers_dict[line]
            # fit the spectrum within +/- 700 km/s of the line center -- follows method in h2_search.py 
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
            # now fit a Gaussian to this H2 line in the average 1D spectrum within the H2-detected region
            # use the continuum rms as surface brightness uncertainty estimate in the fitting
            err_ext = np.ones(whfit.sum()) * cont_rms
            fit_gauss_best = least_squares(residuals, init, bounds=(low, up),
                                           args=(wave_orig[whfit], spec_ext[whfit],
                                                 err_ext, [line_center]))
            # calculate the surface brightness of the line -- need to convert sigma_v to sigma_wave
            # and put back the factor of 1e-4 removed to get nice numbers on y axis. result is in erg/cm^2/s/Sr
            line_sb = np.sqrt(2 * np.pi) * fit_gauss_best.x[2] / 3e5 * line_center * fit_gauss_best.x[3] * 1e-4
            print('H2 {} line intensity: {:.2e}'.format(line.split()[1], line_sb))
            print('H2 {} amplitude {:.1e} = {:.1f}x Continuum RMS'.format(line.split()[1], fit_gauss_best.x[-1],
                                                                          (fit_gauss_best.x[-1] / cont_rms)))
            # calculate uncertainty from fit
            J = fit_gauss_best.jac
            cov = np.linalg.inv(J.T.dot(J))
            stdev = np.sqrt(np.diagonal(cov))
            frac_errs = stdev / fit_gauss_best.x
            line_sb_err = line_sb * np.sqrt((frac_errs[2]) ** 2 + (frac_errs[3]) ** 2)

            # calculate the significance of this line intensity measurement
            nsigma = line_sb / line_sb_err
            print('Intensity uncertainty: {:.2e}'.format(line_sb_err))
            print('From fit uncertainties, this is a {:.1f}-sigma detection'.format(nsigma))
            # consider less than a 3 sigma detection to be a limit
            if nsigma < 3.0:
                print('3-sigma upper limit: {:.2e}'.format(line_sb_err * 3))
                # add the 3-sigma upper limit to the intensity dictionary for modeling below
                intensity[keys_dict[annotate_names[ii]]] = line_sb_err * 3
                intensity_err[keys_dict[annotate_names[ii]]] = line_sb_err
            if nsigma > 3.0:
                # if the line is significantly detected, overplot it on the observed spectrum and annotate the panel
                model = gaussian_fit_bkg(fit_gauss_best.x, wave_orig[whfit],
                                         [line_center])  
                ax.plot(wave[whfit], model, color='C4', label='Gaussian fit', linewidth=1, drawstyle='steps-mid')
                ax.fill_between(wave[whfit], np.zeros(len(wave[whfit])), model, color='C4', alpha=0.2, step='mid')
                ax.legend(frameon=False, fontsize=7, loc='upper left', handlelength=1.0)
                # add the measured intensity to the dictionary for modeling below
                intensity[keys_dict[annotate_names[ii]]] = line_sb
                intensity_err[keys_dict[annotate_names[ii]]] = line_sb_err
                ax.text(0.05, 0.25,
                        r'S(1) intensity=%.1f$\pm$%.1f$\times10^{-7}$erg/s/cm$^2$/sr' % (line_sb / 1e-7,
                                                                                               line_sb_err / 1e-7),
                        fontsize=7, color='k', transform=ax.transAxes)
                ax.text(0.05, 0.2,
                        r'%.1f-$\sigma$ detection' % (line_sb / line_sb_err),
                        fontsize=7, color='k', transform=ax.transAxes)
