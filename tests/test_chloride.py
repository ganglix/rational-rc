import unittest
from rational_rc.chloride import ChlorideModel, load_df_D_RCM, C_crit_param, C_eqv_to_C_S_0
import pandas as pd
import rational_rc.math_helper as mh
import numpy as np

class TestChloride(unittest.TestCase):

    def setUp(self):
        # Initialize test data
        self.x = 40  # [mm]
        self.t = 10  # [year]
        self.pars = DummyPars()

    def test_load_df_D_RCM(self):
        # Call the load_df_D_RCM function
        df = load_df_D_RCM()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 6)
        self.assertEqual(len(df.columns), 4)
        self.assertEqual("wc_eqv" , df.index.name)
        self.assertTrue("CEM_I_42.5_R" in df.columns)
        self.assertTrue("CEM_I_42.5_R+FA" in df.columns)
        self.assertTrue("CEM_I_42.5_R+SF" in df.columns)
        self.assertTrue("CEM_III/B_42.5" in df.columns)

    def test_chloride_model(self):
        # Test ChlorideModel class

        model = ChlorideModel(self.pars)
        model.run(x = self.x, t = self.t)
        model.postproc(plot=False)

        self.assertAlmostEqual(mh.get_mean(model.C_x_t), 0.8765508071783011, places=1)
        self.assertAlmostEqual(model.pf, 0.5263736537728215, places=1)
        self.assertAlmostEqual(model.beta_factor, -0.30799757180427906, places=1)
        
        # calibration
        # field data at three depth at 10 years, (self.t = 10)
        chloride_content_field = pd.DataFrame()
        chloride_content_field['depth'] = [12.5, 50, 100]  # [mm]
        chloride_content_field['cl'] = np.array([0.226, 0.04, 0.014]) / self.pars.cement_concrete_ratio  # chloride_content[wt.-%/cement]

        #calibrate model to the field chloride content
        model_cal = model.calibrate(self.t, chloride_content_field,print_proc=False, plot=True) 
        self.assertAlmostEqual(mh.get_mean(model_cal.pars.D_RCM_0), 70.57514873083403, places = 0)

# Helper class for testing
class DummyPars:
    def __init__(self):
        self.marine = False
        # 1)marine or coastal
        self.C_0_M = 18.980  # natural chloride content of sea water [g/l]
        
        # 2) de-icing salt (hard to quantify)
        self.C_0_R = 0  # average chloride content of the chloride contaminated water [g/l]
        self.n = 0  # average number of salting events per year [-]
        self.C_R_i = 0  # average amount of chloride spread within one spreading event [g/m2]
        self.h_S_i = 1  # amount of water from rain and melted snow per spreading period [l/m2]
        self.C_eqv_to_C_S_0 = C_eqv_to_C_S_0  # imported correlation function for chloride content from solution to concrete
        
        self.exposure_condition = 'splash'
        self.exposure_condition_geom_sensitive = True
        self.T_real = 273 + 25  # averaged ambient temperature[K]

        self.x_a = 10.
        self.x_h = 10.
        self.D_RCM_test = 'N/A'
        self.concrete_type = 'Portland cement concrete'
        self.cement_concrete_ratio = 300. / 2400.
        self.C_max_user_input = None
        self.C_max_option = 'empirical'
        self.C_0 = 0
        self.C_crit_distrib_param = C_crit_param()  # critical chloride content import from Chloride module 0.6 wt.% cement (mean value)
        self.option = DummyOption()

class DummyOption:
    def __init__(self):
        self.choose = True
        self.cement_type = 'CEM_I_42.5_R+SF'
        self.wc_eqv = 0.4  # equivalent water/binder ratio
        self.df_D_RCM_0 = load_df_D_RCM()


if __name__ == '__main__':
    unittest.main()
