# LeoPH2MRS
Code to analyze the H2 rotational lines detected in JWST MIRI-MRS observations of Leo P, presented in Telford et al. (2025, Nature, submitted).

Two Python scripts perform the following calculations on the reduced MRS data:

h2_search.py -- fit a Gaussian centered on the H2 S(1) 17$\mu$m line to the spectrum in each spaxel (each of 4 Ch3 tiles in the 2x2 mosaic covering Leo P is treated separately); plot and report the spaxels where the line is detected at > 3-sigma; plot and report the location on which the congtiguously detected spaxels are centered

h2_excitation_fit.py -- extract the total 1-D spectrum within the only contiguously detected region found above; fit Gaussians centered on the S(1), S(2), and S(3) lines in the integrated spectra from the Ch3 and Ch2 cubes and use best-fit parameters to calculate line intensities; determine the range of single-temperature models and associated column densities allowed by the data

Dependencies: numpy, matplotlib, scipy, astropy

Questions and feedback can be directed to Grace Telford (grace.telford@princeton.edu)