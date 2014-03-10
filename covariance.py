import numpy as np
from scipy import  ndimage
from pandas import Series
from preprocessing import bigfish, threshold


def moment(img, i, j):
    """Utility function called by inertial_axes. See that function, below,
    for attribution and usage."""
    nrows, ncols = img.shape
    y, x = np.mgrid[:nrows, :ncols]
    return (img * x**i * y**j).sum()


def inertial_axes(img): 
    """Calculate the x-mean, y-mean, and cov matrix of an image.
    Parameters
    ----------
    img: ndarray
    
    Returns
    -------
    xbar, ybar, cov (the covariance matrix)

    Attribution
    -----------
    This function is based on a solution by Joe Kington, posted on Stack
    Overflow at http://stackoverflow.com/questions/5869891/
    how-to-calculate-the-axis-of-orientation/5873296#5873296
    """
    normalization = img.sum()
    m10 = moment(img, 1, 0)
    m01 = moment(img, 0, 1)
    x_bar = m10 / normalization 
    y_bar = m01 / normalization
    u11 = (moment(img, 1, 1) - x_bar * m01) / normalization
    u20 = (moment(img, 2, 0) - x_bar * m10) / normalization
    u02 = (moment(img, 0, 2) - y_bar * m01) / normalization
    cov = np.array([[u20, u11], [u11, u02]])
    return x_bar, y_bar, cov


def orientation(cov):
    """Compute the orientation angle of the dominant eigenvector of
    a covariance matrix.

    Parameters
    ----------
    cov: 2x2 array

    Returns
    -------
    angle in radians
    """
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvec = eigvecs[eigvals.argmax()]
    return np.arctan2(eigvec[1], eigvec[0])


def analyze(frame, angle_only=True, plot=False):
    """Find a nanowire in a frame and return its orientation angle
    in degrees.

    Note
    ----
    This convenience function wraps several other functions with detailed
    docstrings. Refer to them for more information.

    Parameters
    ----------
    frame: image array
    angle_only: If True (default), return angle in degrees. If False,
       return x_bar, y_bar, cov -- the C.O.M. and the covariance matrix.
    plot: False by default. If True, plot principle axes over the ROI.
    """
    roi = bigfish(threshold(frame))
    blurred = ndimage.gaussian_filter(frame[roi].astype('float'), 3)
    masked = np.where(threshold(blurred, -0.5),
                     blurred, np.zeros_like(blurred))
    results = inertial_axes(masked)
    if plot:
        import mr.plots
        mr.plots.plot_principal_axes(frame[roi], *results)
    if angle_only:
        return np.rad2deg(orientation(results[2]))
    else:
        return results


def batch(frames):
    """Track the orientation of a wire through many frames.

    Parameters
    ----------
    frames : an iterable, such as a list of images or a mr.Video
        object

    Returns
    -------
    Series of angles in degrees, indexed by frame
    """
    count = frames.count
    data = Series(index=range(1, count + 1))
    for i, img in enumerate(frames):
        data[i + 1] = analyze(img)
    data = data.dropna() # Discard unused rows.
    return data


def periodic_shift(data, shift, period=180):
    return np.mod(data + shift, period) - shift


def shift_ref_frame(data):
    """Choose a cut that avoids splitting the range of observations.

    Parameters
    ----------
    data : array of angles in degrees, revised in place

    Returns
    -------
    None   
    """
    trial_shifts = np.linspace(0, 360, 360)
    spans = np.array([np.ptp(periodic_shift(data, s)) for s in trial_shifts])
    good_shift = trial_shifts[spans.argmin()]
    print good_shift
    return periodic_shift(data, good_shift)
