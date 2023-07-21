import unittest
import numpy as np
import pandas as pd
import rational_rc.math_helper as mh
from rational_rc.carbonation import carb_depth, load_df_R_ACC, CarbonationModel

class TestCarbonation(unittest.TestCase):

    def setUp(self):
        # Initialize test data
        self.t = 50.0
        self.pars = DummyPars()

    def test_load_df_R_ACC(self):
        # Test load_df_R_ACC function
        df = load_df_R_ACC()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 6)
        self.assertEqual(len(df.columns), 4)
        self.assertEqual("wc_eqv" , df.index.name)
        self.assertTrue("CEM_I_42.5_R" in df.columns)
        self.assertTrue("CEM_I_42.5_R+FA" in df.columns)
        self.assertTrue("CEM_I_42.5_R+SF" in df.columns)
        self.assertTrue("CEM_III/B_42.5" in df.columns)

    def test_carbonation_model(self):
        # Test CarbonationModel class
        model = CarbonationModel(self.pars)
        model.run(self.t)
        model.postproc()
        self.assertAlmostEqual(mh.get_mean(model.xc_t), 23.977163903904266, places=1)
        self.assertAlmostEqual(model.pf, 0.0003019205953625868, places=4)
        self.assertAlmostEqual(model.beta_factor, 3.4636404831869974, places=1)
        
        # calibration
        # field data: field carbonation after 20 years, mean=30, std=5
        carb_depth_field = np.array([32.91367329, 28.22985874, 33.39839032, 
                            34.21919435, 33.75538838, 20.50077202,
                            32.43856083, 14.84571027, 31.71301891, 
                            33.4092247,  23.37698412, 30.16002273])
  # mm 
        model = model.calibrate(20, carb_depth_field, print_out=False)
        self.assertAlmostEqual(mh.get_mean(model.xc_t), 29.12743701174534, places=0)
        self.assertAlmostEqual(mh.get_std(model.xc_t), 6.109273222336977, places=0)


# Helper class for testing
class DummyPars:
    def __init__(self):
        self.cover_mean = 50.0
        self.cover_std = 5.0
        self.RH_real = 60.0
        self.t_c = 28
        self.x_c = 0.008
        self.ToW = 2 / 52.
        self.p_SR = 0.0
        self.C_S_emi = 0.0
        self.option = DummyOption()

class DummyOption:
    def __init__(self):
        self.choose = False
        self.cement_type = 'CEM_I_42.5_R+SF'
        self.wc_eqv = 0.6
        self.df_R_ACC = load_df_R_ACC()
        self.plot = True

if __name__ == '__main__':
    unittest.main()
