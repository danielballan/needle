import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.stats import linregress
from pandas import Series
from preprocessing import bigfish, threshold


def _gaussian(x, A, sigma, x0):
    return A*np.exp(-(x - x0)**2/(2*sigma))


def fit_row(x, row, guess):
    "Fit a Gaussian to a single row of pixels."
    try:
        popt, pcov = curve_fit(_gaussian, x, row, p0=guess)
    except RuntimeError:
        return np.nan*np.empty((3,)), np.nan*np.empty((3, 3))
    return popt, pcov


def fit_rows(image, guess_sigma=5.):
    "Fit a Gaussian to each row of pixels. Return best fits, fit covariances."
    x = np.arange(image.shape[1])
    guess = [image.max(), guess_sigma, image.shape[1] / 2.]
    fits = np.empty((image.shape[0], 3))
    for i, row in enumerate(image):
        fit, cov = fit_row(x, row, guess)
        guess = fit
        fits[i] = fit
    return fits


def infer_angle_from_centers(x0):
    "Fit a line to the center positions in each row and compute its angle."
    x0 = Series(x0).dropna()
    len_x0 = len(x0)
    if len_x0 < 3:
        raise ValueError("Only {0} points were successfully fit.".format(len_x0))
    y = x0.index.values.astype(int)
    slope, intercept, r_value, p_value, std_err = linregress(x0, y)
    angle = np.arctan(slope)
    print slope, np.rad2deg(angle)
    return angle


def analyze(image, guess_sigma=3.):
    roi = bigfish(threshold(image))
    blurred = ndimage.gaussian_filter(image[roi].astype('float'), 3)
    masked = np.where(threshold(blurred, -0.5),
                      blurred, np.zeros_like(blurred))
    fits = fit_rows(blurred, guess_sigma)
    angle = infer_angle_from_centers(fits[:, 2])
    return np.rad2deg(angle)
