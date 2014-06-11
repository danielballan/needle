import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
from pandas import Series
from skimage import transform


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
    angle = np.pi/2 - np.arctan(slope)
    print np.rad2deg(angle)
    return angle


class ConvergenceError(Exception):
    pass


def analyze(image, guess_sigma=3., max_iterations=20):
    """Discern the orientation of an elongated object in an image.

    Fit a Gaussian to each row of image, and fit a line along their
    centers. Rotate the image and perform the fit again, iteratively.

    Parameters
    ----------
    image : image array
    guess_sigma : initial guess for Gaussian width of wire
    max_iterations : number of times to rotate image retry fit, 20 by default

    Returns 
    -------
    Series with 'x' center, 'y' center, and 'angle' in radians

    Note
    ----
    This technique does not find the center of the wire.
    The center x, y is just the center of the ROI.
    """
    fits = fit_rows(image, guess_sigma)
    x0 = fits[:, 2]
    angle = infer_angle_from_centers(x0)
    total_angle = angle
    i = 0
    while np.abs(angle) > 5:
        image = transform.rotate(image, -angle)
        fits = fit_rows(image, guess_sigma)
        angle = infer_angle_from_centers(fits[:, 2])
        total_angle += angle
        if i > max_iterations:
            raise ConvergenceError(
                "After {0} consecutive rotations, the image could not be "
                "aligned to the vertical.".format(max_iterations))
        i += 1
    return Series(total_angle, image.shape[0] // 2, image.shape[1] // 2)
