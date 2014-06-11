import numpy as np
from skimage import filter
from scipy import ndimage

def threshold(image, sigma=3):
    """Threshold a grayscale image based on the mean and std brightness.

    Parameters
    ----------
    image: ndarray
    sigma: float, default 3.0
        minimum brightness in terms of standard deviations above the mean
    """
    mask = image > (image.mean() + sigma*image.std())
    return mask

def primary_object(mask, padding=0.03):
    """Identify the largest connected region and return the roi. 

    Parameters
    ----------
    mask: binary (thresholded) image
    padding: fractional padding of ROI (default 0.02)

    Returns
    -------
    padded_roi: a tuple of slice objects, for indexing the image
    """
    label_im, nb_labels = ndimage.label(mask)
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    big_label = sizes.argmax() # the label of the largest connection region
    roi = ndimage.find_objects(label_im==big_label)[0]
    padded_roi = pad_roi(roi, padding, mask.shape)
    return padded_roi

def pad_roi(roi, padding, img_shape):
    "Pad x and y slices, within the bounds of img_shape."
    s0, s1 = roi # slices in x and y
    p = int(np.max(img_shape)*padding)
    new_s0 = slice(np.clip(s0.start - p, 0, img_shape[0] - 1),
                   np.clip(s0.stop + p, 0, img_shape[0] - 1))
    new_s1 = slice(np.clip(s1.start - p, 0, img_shape[1] - 1),
                   np.clip(s1.stop + p, 0, img_shape[1] - 1))
    return new_s0, new_s1


def preprocess(image, gaussian_sigma=1, mask_threshold=-0.5):
    roi = primary_object(threshold(image))
    image = image[roi].astype(float)
    blurred = filter.gaussian_filter(image, gaussian_sigma)
    if mask_threshold:
        masked = np.where(threshold(blurred, mask_threshold),
                         blurred, np.zeros_like(blurred))
        return masked
    else:
        return blurred 
