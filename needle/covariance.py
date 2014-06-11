import numpy as np
from pandas import Series


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
    principal_moment = eigvecs[eigvals.argmax()]
    return np.arctan2(principal_moment[1], principal_moment[0])


def analyze(image):
    """Discern the orientation of an elongated object in an image.

    Compute the image's covariance matrix ("inertial tensor" if brightness
    is mass). Find the direction of the principle moment of inertia.

    Parameters
    ----------
    image : image array

    Returns
    -------
    DataFrame with 'x' center, 'y' center, and 'angle' in radians
    """
    results = inertial_axes(image)
    results[2] = orientation(results[2])
    return Series(results, columns=['x', 'y', 'angle'])
