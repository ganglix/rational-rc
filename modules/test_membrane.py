import unittest
import matplotlib.pyplot as plt
from membrane import MembraneModel

class TestMembraneModel(unittest.TestCase):

    def setUp(self):
        self.pars = DummyPars()

    def test_MembraneModel(self):
        model = MembraneModel(self.pars)
        model.run(10)  # 10 years
        model.postproc(plot=False)

        self.assertAlmostEqual(model.pf, 0.050, places=2)
        self.assertAlmostEqual(model.beta_factor, 1.638666969042218, places=1)
        
        model_cal = model.calibrate(self.pars.membrane_age_field, 
                                    self.pars.membrane_failure_ratio_field)
        model_cal.run(10)  # 10 years
        model_cal.postproc(plot=False)
        self.assertAlmostEqual(model_cal.pf, 0.17795324587799488, places=1)
        self.assertAlmostEqual(model_cal.beta_factor, 0.9218308896882579, places=2)


class DummyPars:
    def __init__(self):
        # product information
        self.life_product_label_life = 10  # year, defined as 95% confident non-failure
        self.life_std = 0.2 * self.life_product_label_life # assume if not known, calibrate later for real service conditions
        self.life_confidence = 0.95

        # calibration data (if available)
        # field survey result
        self.membrane_failure_ratio_field = 0.01
        self.membrane_age_field = 5  # [year]


if __name__ == "__main__":
    unittest.main()
