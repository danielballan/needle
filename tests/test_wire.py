import unittest
import os
import numpy as np
from numpy.testing import assert_allclose
from skimage import draw, transform, filter
import trackwire
from trackwire import covariance
from trackwire import gaussians
from trackwire import preprocessing
from trackwire.preprocessing import preprocess

data_dir = os.path.join(os.path.dirname(trackwire.__file__), '..', 'tests', 'data')

oblique = np.load(os.path.join(data_dir, 'oblique_frame.npy'))
vertical = np.load(os.path.join(data_dir, 'vertical_frame.npy'))
horizontal = np.load(os.path.join(data_dir, 'horizontal_frame.npy'))


def sim_wire(angle, gaussian_sigma=1, noise_level=0, L=100):
    "Draw a wire with blur and optional noise."
    # Noise level should be from 0 to 1. 0.02 is reasonable.
    shape = (L, L)

    a = np.zeros(shape, dtype=np.uint8)
    a[draw.ellipse(L//2, L//2, L//24, L//4)] = 100  # horizontal ellipse
    a = filter.gaussian_filter(a, gaussian_sigma)
    b = transform.rotate(a, angle)
    b += noise_level*np.random.randn(*shape)
    return b


class BaseTestWire(object):

    def compare(self, angle, atol, gaussian_sigma=1, mask_threshold=False,
                noise_level=0, L=100):
        image = preprocess(sim_wire(angle), gaussian_sigma, 
                           mask_threshold)
        actual = self.analyze(image)
        assert_allclose(actual, angle, atol=atol)

#    def test_real_oblique_wire(self):
#        image = preprocess(oblique, **self.kwargs)
#        expected = 125  # measured manually in ImageJ
#        assert_allclose(self.analyze(image), expected, atol=3)

#    def test_real_vertical_wire(self):
#        expected = 91
#        assert_allclose(self.analyze(vertical), expected, atol=3)

#    def test_real_horizontal_wire(self):
#        expected = 3
#        assert_allclose(self.analyze(horizontal), expected, atol=3)


    def test_clean_wire_first_quadrant(self):
        first_quadrant = [1, 10, 30, 50, 80, 89]
        [self.compare(angle, atol=2, **self.kwargs) for angle in first_quadrant]

    def test_noisy_wire_first_quadrant(self):
        first_quadrant = [1, 10, 30, 50, 80, 89]
        [self.compare(angle, noise_level=0.01, atol=2, **self.kwargs)
         for angle in first_quadrant]


class TestCovariance(BaseTestWire, unittest.TestCase):
    def setUp(self):
        self.kwargs = {'gaussian_sigma': 1, 'mask_threshold': 1}

    def analyze(self, image, **kwargs):
        return covariance.analyze(image, **kwargs)


class TestGaussians(BaseTestWire, unittest.TestCase):
    def setUp(self):
        self.kwargs = {'gaussian_sigma': 2, 'mask_threshold': False}

    def analyze(self, image, **kwargs):
        return gaussians.analyze(image, **kwargs)
