"""Fast physics and tensor-shape smoke tests (unittest, no pytest needed)."""
import unittest
import numpy as np
from main import FS, C, D, simulate_pair, gcc_phat_stft, tau_to_angle
from lesson06_6 import extract_ipd

class TestSSL(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1234)
        self.source = self.rng.normal(size=5120)

    def test_geometry_signs(self):
        for angle in [-30, 0, 30]:
            a,b,true_tau = simulate_pair(angle, self.source, snr_db=30, seed=42)
            expected = -(D/C)*np.sin(np.deg2rad(angle))
            self.assertAlmostEqual(true_tau, expected, places=10)
            self.assertAlmostEqual(tau_to_angle(true_tau), angle, places=9)

    def test_estimation_near_known_angle(self):
        a,b,_ = simulate_pair(30, self.source, snr_db=30, seed=42)
        tau,*_ = gcc_phat_stft(a,b)
        self.assertLess(abs(tau_to_angle(tau)-30), 3.0)

    def test_ipd_feature_shape(self):
        a,b,_ = simulate_pair(-25, self.source, snr_db=20, seed=7)
        ipd = extract_ipd(a,b)
        self.assertEqual(ipd.shape, (48,))
        self.assertEqual(ipd.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(ipd)))

if __name__ == '__main__':
    unittest.main()
