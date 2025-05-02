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

# now that we have intensity measurements/limits, we can test what range of temperatures and column densities are allowed
# first, define some constants (many from Roeuff+ 2019; see Table 2 in Telford+ 2025)
Eu_over_k_dict = {'H200S1': 1015.1, 'H200S2': 1681.6, 'H200S3': 2503.7}  # K
A_dict = {'H200S1': 47.6e-11, 'H200S2': 275.5e-11, 'H200S3': 983.6e-11}  # 1/s
waves_dict = {'H200S1': 17.0348, 'H200S2': 12.2786, 'H200S3': 9.6649}  # micron
planck_const = 6.626e-27  # erg s
c = 3e10  # cm/s
cm_per_micron = 1e-4


def partition_func(temp):
    # temperature in K; see Herbst+ (1996)
    return 0.0247 * temp / (1 - np.exp(-6000. / temp))


def coldens_line(line_name):
    # calculate the column density in the upper level (Equation 1 in Telford+ 2025)
    numerator = 4 * np.pi * (waves_dict[line_name] * cm_per_micron) * intensity[line_name]
    denominator = planck_const * c * A_dict[line_name]
    return numerator / denominator


def g_line(line_name):
    # calculate the statistical weight of the upper level of each transition
    j = float(line_name[-1]) + 2
    return (2 * (j % 2) + 1) * (2 * j + 1)


# calculate observed column densities from fluxes
lnratios = np.array([])
lnratios_unc = np.array([])
energies = np.array([])
for line in intensity.keys():
    logratio = np.log10(coldens_line(line) / g_line(line))
    # take ln(column density / statistical weight)
    lnratios = np.append(lnratios, np.log(coldens_line(line) / g_line(line)))  
    energies = np.append(energies, Eu_over_k_dict[line])
    lnratios_unc = np.append(lnratios_unc, intensity_err[line] / intensity[line])  # this seems weird but OK...
# reverse the order of these lists to be compatible with the below code
lnratios = lnratios[::-1]
lnratios_unc = lnratios_unc[::-1]
energies = energies[::-1]

# fit a line to the ln(N/g) vs. E points -- this is just a reference point for scaling initial slope/N for every T we test below
coeffs = np.polyfit(energies, lnratios, 1, w=lnratios_unc ** -1)
p = np.poly1d(coeffs)
plotfit = np.linspace(0, 3500, 1001)
# plot the measured (or limits on) ln(N_u/g_u) vs. transition energy in K in the final panel of the figure
ax = ax_fit
uplims = np.array([0, 1, 1], dtype=bool)
ax.errorbar(energies, lnratios, lnratios_unc, fmt='ko', ms=4, elinewidth=1,
            capsize=2, uplims=uplims, label='MIRI-MRS data', zorder=5)
# annotate the 3 points
labels = [r'H$_2$ S(1)', r'H$_2$ S(2)', r'H$_2$ S(3)']
for ii in range(3):
    if ii == 0:
        ax.text(energies[ii] + 60, lnratios[ii] + 0.03, labels[ii], ha='left', va='bottom', fontsize=7)
    else:
        ax.text(energies[ii] + 40, lnratios[ii] + 0.03, labels[ii], ha='left', va='bottom', fontsize=7)

# now, from that initial fit, we're going to iterate over a wide range of temperatures and find the 
# the min/best/max column densities allowed for each temperature
Ts_that_work = np.array([])
Ns_implied = np.array([])
Ts_test = np.arange(80, 1001, 10)
line_values = np.empty([len(Ts_test) * 3, len(plotfit)])
mask = []
# find the range of T, N allowed within 1-sigma of the S(1) intensity and 3-sigma upper limit on S(3) [more constraining than S(2)]
for ii, T_fix in enumerate(Ts_test):
    # this is a line with the same y-intercept from the initial fit above, but slope equal to -1/(this temperature)
    p_new = np.poly1d([-1. / T_fix, p[0]])
    # find the energy sample closest to S(1)
    wh = np.argmin(np.abs(plotfit - Eu_over_k_dict['H200S1']))  
    # calculate vertical offset to match the max, best, and min S(1)
    scale_factors = np.array([lnratios[0] + lnratios_unc[0], lnratios[0],  
                              lnratios[0] - lnratios_unc[0]]) - p_new(plotfit[wh])
    # use these vertical offsets to calculate the associated column densities with each 
    Ns_new = np.exp(p[0] + scale_factors) * 0.0247 * T_fix / (1 - np.exp(-6000 / T_fix))
    # now, check the column density predicted for S(3)
    wh_s3 = np.argmin(np.abs(plotfit - Eu_over_k_dict['H200S3']))
    for jj in range(3):
        # these are the power laws to plot associated with the min, best, and max column density given S(1) at this temperature
        line_values[ii * 3 + jj, :] = p_new(plotfit) + scale_factors[jj]
        if p_new(plotfit[wh_s3]) + scale_factors[jj] < lnratios[2]:
            # ONLY add this temperature/column density pair that works for S(1) to the lists of values that are allowed if
            # they are also consistent with the 3-sigma limit on S(3)
            Ts_that_work = np.append(Ts_that_work, T_fix)
            Ns_implied = np.append(Ns_implied, Ns_new[jj])
            mask.append(True)
        else:
            mask.append(False)
# get rid of any power-law functions that are NOT consistent with both S(1) and S(3) constraints
line_values = line_values[mask, :]
ax.set_xlim(0, 3200)
ax.set_ylim(31.5, 38.5)
# get nicely formatted column densities for the minimum/maximum overall allowed T and N values
pow_max = float(str(np.max(Ns_implied))[-2:])
pow_min = float(str(np.min(Ns_implied))[-2:])
num_max = np.max(Ns_implied) / 10 ** pow_max
num_min = np.min(Ns_implied) / 10 ** pow_min
# finally, plot the power laws for these limiting parameters along with the data in the final panel
ax.plot(plotfit, line_values[0, :], color='C0', label=r'$T$=%.0f K, ' % np.min(Ts_that_work) +
        r'$N$=%.1f$\times10^{%.0f}$cm$^{-2}$' % (num_max, pow_max))
ax.plot(plotfit, line_values[-1, :], color='C3', label=r'$T$=%.0f K, ' % np.max(Ts_that_work) +
        r'$N$=%.1f$\times10^{%.0f}$cm$^{-2}$' % (num_min, pow_min))
ax.legend(frameon=False, fontsize=7, markerfirst=False, handletextpad=0.5, handlelength=1.0)

# from these limits, calculate implied warm H2 mass
solid_angle = np.pi * (0.668/2) ** 2  # 1 resolution element at 17 micron in arcsec^2
cm_per_arcsec = 1.6 * 3.086e+24 * 4.848e-6  # distance (Mpc) * cm/Mpc * rad/arcsec

number_H2_new = np.min(Ns_implied) * solid_angle * cm_per_arcsec ** 2
m_H2_new = number_H2_new * 2 * 1.674e-24 / 1.989e33
print('\n\nGiven maximum allowed T of {:.0f} K, warm H2 mass is at least {:.2f} Msun'.format(np.max(Ts_that_work),
                                                                                         m_H2_new))
number_H2_max = np.max(Ns_implied) * solid_angle * cm_per_arcsec ** 2
m_H2_max = number_H2_max * 2 * 1.674e-24 / 1.989e33
print('Assuming T = {} K, warm H2 mass can be up to {:.2f} Msun'.format(np.min(Ts_test), m_H2_max))
