import unittest
import numpy as np
from corrosion import CorrosionModel, SectionLossModel

class TestCorrosion(unittest.TestCase):
    def setUp(self):
        self.pars = DummyCorrPars()

    def test_CorrosionModel(self):
        # initialize and run model
        model_corr = CorrosionModel(self.pars)
        model_corr.run()

        # result 
        self.assertAlmostEqual(model_corr.icorr.mean(), 0.006407338781, places=3)
        self.assertAlmostEqual(model_corr.x_loss_rate.mean(), 0.00742047, places=3)
        self.model_corr = model_corr

    def test_SectionLossModel(self):
        # A solved model_corr
        model_corr = CorrosionModel(self.pars)
        model_corr.run()
        
        # time steps
        t_lis = np.linspace(0, 365*100 , 100)  

        # Given probability of active corrosion with time, and the section loss  (determined by membrane, carbonation, chloride module)
        # dummy data used for this example
        pf_lis = np.linspace(0,1,len(t_lis))**5

        # prepare Param object for section loss object
        self.pars.x_loss_rate = model_corr.x_loss_rate.mean()     # mm/year mean section loss rate from the corrosion model
        self.pars.p_active_t_curve = (pf_lis, t_lis)              # use dummy data for this example

        # critical section loss from the external structural analysis
        self.pars.x_loss_limit_mean = 0.5         # mm
        self.pars.x_loss_limit_std = 0.5 * 0.005  # mm

        # initialize section loss model object
        model_sl = SectionLossModel(self.pars)
        
        model_sl.run(t_end = 70)
        model_sl.postproc(plot=False)

        # results
        self.assertAlmostEqual(model_sl.pf, 0.781684, places=2)
        self.assertAlmostEqual(model_sl.beta_factor, -5.814361, places=1)


class DummyCorrPars:
    def __init__(self):
        # geometry and age
        self.d = 0.04  # cover depth [m]
        self.t = 3650  # age[day]

        # concrete composition
        self.cement_type = 'Type I'
        self.concrete_density = 2400 #kg/m^3
        self.a_c = 2        # aggregate(fine and coarse)/cement ratio
        self.w_c = 0.5      # water/cement ratio
        self.rho_c= 3.1e3   # density of cement particle [kg/m^3]
        self.rho_a= 2600.   # density of aggregate particle(fine and coarse) range 2400-2900 [kg/m^3]


        # concrete condition
        self.epsilon = 0.25     # porosity of concrete
        self.theta_water = 0.12 # volumetric water content
        self.T = 273.15+25      # temperature [K]


if __name__ == '__main__':
    unittest.main()
