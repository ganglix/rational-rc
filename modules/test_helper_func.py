import unittest
import helper_func as hf
import numpy as np
from scipy import stats


class TestHelperFunc(unittest.TestCase):
    # setup reusable materials to test
    def setUp(self):
        self.arr_mix = np.array([1, 2, np.nan, 3, 4, np.nan])

    def tearDown(self):
        pass

    # tester
    def test_dropna(self):
        # compare array
        self.assertCountEqual(hf.dropna(self.arr_mix), np.array([1, 2, 3, 4]))

    def test_Get_mean(self):
        self.assertEqual(hf.Get_mean(self.arr_mix), 10 / 4)

    def test_Get_std(self):
        self.assertEqual(hf.Get_std(self.arr_mix), 1.118033988749895)

    def test_Normal_custom(self):
        res1 = hf.Normal_custom(1, 0.1)

        self.assertTrue(np.mean(res1) > 1 * 0.95)
        self.assertTrue(np.mean(res1) < 1 * 1.05)
        self.assertTrue(np.std(res1) > 0.1 * 0.95)
        self.assertTrue(np.std(res1) < 0.1 * 1.05)

        res2 = hf.Normal_custom(0, 1, non_negative=True, plot=False)
        self.assertTrue((res2 >= 0).all())
        self.assertTrue(np.mean(res2) > 1 * 2 ** 0.5 / np.pi ** 0.5 * 0.95)
        self.assertTrue(np.mean(res2) < 1 * 2 ** 0.5 / np.pi ** 0.5 * 1.05)
        self.assertTrue(np.std(res2) > ((1 ** 2 * (1 - 2 / np.pi)) ** 0.5 * 0.95))
        self.assertTrue(np.std(res2) < ((1 ** 2 * (1 - 2 / np.pi)) ** 0.5 * 1.05))

    def test_Beta_custom(self):
        res1 = hf.Beta_custom(1, 0.1, 0, 2, plot=False)
        self.assertTrue((np.std(res1) <= 2).all())
        self.assertTrue((np.std(res1) >= -2).all())

        self.assertTrue(np.mean(res1) > 1 * 0.95)
        self.assertTrue(np.mean(res1) < 1 * 1.05)
        self.assertTrue(np.std(res1) > 0.1 * 0.95)
        self.assertTrue(np.std(res1) < 0.1 * 1.05)

    def test_interp_extrap_f(self):
        x1 = np.linspace(0, 1, 5)
        y1 = np.linspace(0, 2, 5)
        xfind = np.array([0.5, 0.7])
        res1 = hf.interp_extrap_f(x1, y1, xfind, plot=False)
        np.testing.assert_almost_equal(res1, np.array([1.0, 1.4]))

        x2 = np.linspace(0, 1, 3)
        y2 = np.linspace(0, 2, 3)
        res2 = hf.interp_extrap_f(x2, y2, xfind, plot=False)
        np.testing.assert_almost_equal(res2, np.array([1, 1.4]))

    def test_find_similar_group(self):
        lis = [1, 1.1, 2, 3]
        self.assertCountEqual(
            hf.find_similar_group(lis, similar_group_size=2), [1, 1.1]
        )

    def test_sample_integral(self):
        Y35 = np.array(
            [
                [1, 1.1, 1.01, 1.001, 1.0001],
                [2, 2.1, 2.01, 2.001, 2.0001],
                [3, 3.1, 3.01, 3.001, 3.0001],
            ]
        )
        x5 = np.array([1, 2, 3, 4, 5])
        with self.assertRaises(Exception):
            hf.sample_integral(Y35, x5)

        x2 = np.array([1, 2, 3])
        np.testing.assert_almost_equal(
            hf.sample_integral(Y35, x2), np.array([4.0, 4.2, 4.02, 4.002, 4.0002])
        )

    def test_f_solve_poly2(self):
        self.assertTrue(1.0 in hf.f_solve_poly2(1, -2, 1))

    def test_Fit_distrib(self):

        sample = hf.Normal_custom(1, 0.1)

        fit_normal = hf.Fit_distrib(sample, fit_type="normal")
        # is stats.normal instance
        self.assertIsInstance(fit_normal, stats._distn_infrastructure.rv_frozen)
        # value is restored
        self.assertAlmostEqual(fit_normal.mean(), 1, places=2)
        self.assertAlmostEqual(fit_normal.std(), 0.1, places=3)

        fit_kde = hf.Fit_distrib(sample, fit_type="kernel")
        print(fit_kde)
        # is kde instance
        self.assertIsInstance(fit_kde, stats.kde.gaussian_kde)
        # value is restored
        self.assertAlmostEqual(fit_kde.resample(100000).mean(), 1, places=2)
        self.assertAlmostEqual(fit_kde.resample(100000).std(), 0.1, places=2)

    def test_Pf_RS(self):
        sample = hf.Normal_custom(1, 0.3)
        pf_RS, beta_factor, R_distrib, S_kde_fit = hf.Pf_RS(
            R_info=(2, 0.1), S=sample, R_distrib_type="normal", plot=False
        )
        self.assertAlmostEqual(pf_RS, 0.0008761213723265322, places=2)
        self.assertAlmostEqual(beta_factor, 3.153227215906705, places=1)
        self.assertAlmostEqual(R_distrib.mean(), 2, places=3)
        self.assertAlmostEqual(
            S_kde_fit.resample(100000).mean(), sample.mean(), places=2
        )

        pf_RS, beta_factor, R_distrib, S_kde_fit = hf.Pf_RS(
            R_info=(2, 0.5, 0, 8), S=sample, R_distrib_type="beta", plot=False
        )
        self.assertAlmostEqual(pf_RS, 0.036970274667017286, places=2)
        self.assertAlmostEqual(beta_factor, 1.7094971728697723, places=1)
        self.assertAlmostEqual(R_distrib.mean(), 2, places=2)
        self.assertAlmostEqual(
            S_kde_fit.resample(100000).mean(), sample.mean(), places=2
        )

    def test_RS_plot(self):
        # how to test the plot function?
        pass

    def test_find_mean(self):
        std = 10 / 1.64
        cutoff = 30
        self.assertAlmostEqual(
            hf.find_mean(cutoff, std, confidence_one_tailed=0.95),
            40.029595286288796,
            msg="95% one tailed",
        )
        self.assertAlmostEqual(
            hf.find_mean(cutoff, std, confidence_one_tailed=0.90),
            37.81433881429602,
            msg="90% one tailed",
        )


if __name__ == "__main__":
    unittest.main()
