import unittest
import os
import numpy as np
from numpy.testing import assert_allclose
from skimage import draw, transform, filter
import covariance
import gaussians

data_dir = os.path.join(os.path.dirname(covariance.__file__), 'tests', 'data')

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

    def compare(self, angle, atol, gaussian_sigma=1, noise_level=0, L=100):
        actual = self.analyze(sim_wire(angle), **self.kwargs)
        assert_allclose(actual, angle, atol=atol)

#    def test_real_oblique_wire(self):
#        assert_allclose(self.analyze(oblique, **self.kwargs), 53, atol=5)

#    def test_real_vertical_wire(self):
#        assert_allclose(self.analyze(vertical, **self.kwargs), 91, atol=5)

#    def test_real_horizontal_wire(self):
#        assert_allclose(self.analyze(horizontal, **self.kwargs), 3, atol=5)


    def test_clean_wire_first_quadrant(self):
        first_quadrant = [1, 10, 30, 50, 80, 89]
        [self.compare(angle, atol=2) for angle in first_quadrant]

    def test_noisy_wire_first_quadrant(self):
        first_quadrant = [1, 10, 30, 50, 80, 89]
        [self.compare(angle, noise_level=0.01, atol=2)
         for angle in first_quadrant]


class TestCovariance(BaseTestWire, unittest.TestCase):
    def setUp(self):
        self.kwargs = {}

    def analyze(self, image, **kwargs):
        return covariance.analyze(image, **kwargs)


class TestGaussians(BaseTestWire, unittest.TestCase):
    def setUp(self):
        self.kwargs = {'gaussian_sigma': 2}

    def analyze(self, image, **kwargs):
        return gaussians.analyze(image, **kwargs)
