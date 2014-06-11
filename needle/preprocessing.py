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
    padded_roi: a tuple of slice objects
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


def prepare(image, primary_sigma=3, blur_sigma=1, mask_sigma=-0.5):
    """Prepare the image using a sequence of standard preprocessing operations.
    
    1. Identify the primary object (largest connected region).
    2. Crop to that ROI (Region Of Interest) with a margin around the object.
    3. Blur the image and mask out pixels that are away from the object.
    4. Return the ROI slice and the processed ROI image itself.

    Parameters
    ----------
    image : grayscale image array
    primary_sigma : sigma used in initial threshold operation to isolate objects
    blur_sigma : Gaussian blur sigma
    mask_sigma : sigma used in final threshold to crush near-black regions

    Returns
    -------
    a tuple: roi, processed_image
    """

    roi = primary_object(threshold(image, primary_sigma))
    image = image[roi].astype(float)
    blurred = filter.gaussian_filter(image, blur_sigma)
    if mask_sigma:
        masked = np.where(threshold(blurred, mask_sigma),
                         blurred, np.zeros_like(blurred))
        return roi, masked
    else:
        return roi, blurred
