import unittest
from rational_rc.cracking import Cracking_Model
import numpy as np
import rational_rc.math_helper as mh

class TestCracking(unittest.TestCase):
    def setUp(self):
        # Set up the necessary parameters for testing
        self.pars = DummyPars()

        # Create a Cracking_Model instance
        self.model = Cracking_Model(self.pars)
        
    def test_run_deterministic(self):
        # Run the model in deterministic mode
        self.model.run(stochastic=False, plot_deterministic_result = False)
        
        np.testing.assert_array_almost_equal(self.model.epsilon_theta[0, 0:100:20],
                                             np.array([np.nan, 7.99743339e-05, 3.50349022e-05, 2.11879193e-05,1.51800392e-05]),
                                             decimal=6)
        np.testing.assert_array_almost_equal(self.model.sigma_theta[0, 1:100:30],
                                             np.array([0.71194436, 1.5450239,  0.66794578, 0.4256151]),
                                             decimal=1)
        self.assertAlmostEqual(self.model.rust_thickness[0], 7.20151627e-06, places=7)
        self.assertAlmostEqual(self.model.R_c[0,0], 0.00850738, places=4)

    def test_run_stochastic(self):
        # Run the model in stochastic mode
        self.model.run(stochastic=True)

        self.assertAlmostEqual(mh.get_mean(self.model.rust_thickness), 7.20151627e-06, places=7)
        self.assertAlmostEqual(mh.get_mean(self.model.R_c), 0.01035, places=2)

        # Post-process the results
        self.model.postproc()

        self.assertAlmostEqual(mh.get_mean(self.model.crack_length_over_cover), 0.095, places=2)
        

class DummyPars:
    def __init__(self):
        # material properties
        r0_bar_mean = 5e-3          # rebar diameter [m]
        f_t_mean=5.                 # concrete ultimate tensile strength[MPa]
        E_0_mean=32e3               # concrete modulus of elasticity [Mpa]

        x_loss_mean = 12.5e-6*0.6   # rebar section loss, mean [m]
        cover_mean = 4e-2           # cover thickness, mean [m]

        self.r0_bar = mh.normal_custom(r0_bar_mean, 0.1*r0_bar_mean, non_negative=True)
        self.x_loss = mh.normal_custom(x_loss_mean, 0.1*x_loss_mean, non_negative=True)  # or from the corrosion model solution
        self.cover = mh.normal_custom(cover_mean, 0.1*cover_mean, non_negative=True)
        self.f_t = mh.normal_custom(f_t_mean, 0.1*f_t_mean, non_negative=True)
        self.E_0 = mh.normal_custom(E_0_mean, 0.1*E_0_mean, non_negative=True)
        self.w_c = mh.normal_custom(0.5, 0.1*0.6, non_negative=True)
        self.r_v = mh.beta_custom(2.96, 2.96*0.05, 3.3, 2.6)  # rust volumetric expansion rate  2.96 lower 2.6  upper: 3.3


if __name__ == '__main__':
    unittest.main()
