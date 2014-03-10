import unittest
import os
import numpy as np
from numpy.testing import assert_almost_equal
import covariance
import gaussians

data_dir = os.path.join(os.path.dirname(covariance.__file__), 'tests', 'data')


class BaseTestWire(object):
    def setUp(self):
        self.oblique = np.load(os.path.join(data_dir, 'oblique_frame.npy'))
        self.vertical = np.load(os.path.join(data_dir, 'vertical_frame.npy'))
        self.horizontal = np.load(os.path.join(data_dir, 'horizontal_frame.npy'))
       
    def test_oblique_wire(self):
        assert_almost_equal(self.analyze(self.oblique), 53.392, decimal=0)

    def test_vertical_wire(self):
        assert_almost_equal(self.analyze(self.vertical), 91.484, decimal=0)

    def test_horizontal_wire(self):
        assert_almost_equal(self.analyze(self.horizontal), -177.515, decimal=0)


class TestCovariance(BaseTestWire, unittest.TestCase):
    def analyze(self, image):
        return covariance.analyze(image)


class TestGaussians(BaseTestWire, unittest.TestCase):
    def analyze(self, image):
        return gaussians.analyze(image)
