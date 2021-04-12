"""

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from copy import deepcopy
import logging

import helper_func as hf

# logger
# log levels: NOTSET, DEBUG, INFO, WARNING, ERROR, and CRITICAL
LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(
    filename="mylog.log",
    # level=logging.DEBUG,
    format=LOG_FORMAT,
)

logger = logging.getLogger(__name__)
logger.setLevel(
    logging.CRITICAL
)  # set logging level here to work in jupyter notebook where maybe a default setting was there


# model functions
def Carb_depth(t, pars):
    """ Master model function, calculate carbonation depth and the k constant of sqrt of time from all the
    parameters. The derived parameters is also calculated within this funcion. Caution: The pars instance is mutable,
    so a deepcopy of the original instance should be used if the calculation is not intended for "inplace".

    Parameters
    ----------
    t      : time [year]
    pars   : object/instance of wrapper class(empty class)
               a wrapper of all material and environmental parameters deep-copied from the raw data

    Returns
    -------
    out : carbionation depth at the time t [mm]

    Note
    ----
        intermediate parameters calculated and attached to 
        
        pars k_e : environmental function [-] 
        
        k_c : execution transfer parameter [-] 
        
        account for curing measures 
        
        k_t : regression parameter [-] 
        
        R_ACC_0_inv: inverse effective carbonation resistance of concrete(accelerated) [(mm^2/year)/(kg/m^3)] eps_t  : error term [-]
        
        C_S    : CO2 concentration [$kg/m^3$] 
        
        W_t    : weather function [-] 
        
        k      : constant before the sqrt of time(time[year], carbonation depth[mm]) [mm/year^0.5] 
        typical value of k =3~4 for unit mm,year [https://www.researchgate.net/publication/272174090_Carbonation_Coefficient_of_Concrete_in_Dhaka_City]
    """
    pars.t = t
    pars.k_e = k_e(pars)
    pars.k_c = k_c(pars)
    pars.k_t = k_t()
    pars.R_ACC_0_inv = R_ACC_0_inv(pars)
    pars.eps_t = eps_t()
    pars.C_S = C_S(C_S_emi=pars.C_S_emi)
    pars.W_t = W_t(t, pars)

    pars.k = (
        2 * pars.k_e * pars.k_c * (pars.k_t * pars.R_ACC_0_inv + pars.eps_t) * pars.C_S
    ) ** 0.5 * pars.W_t
    xc_t = pars.k * t ** 0.5
    return xc_t


# data import function
def load_df_R_ACC():
    """load the data table of the accelerated carbonation test
    for R_ACC interpolation.

    Parameters
    ----------

    Returns
    -------
    Pandas Dataframe

    Notes
    -----
    w/c 0.45 cemI is comparable to ACC of 3 mm.
    """
    wc_eqv = np.arange(0.35, 0.60 + (0.05 / 2), 0.05)
    df = pd.DataFrame(
        columns=[
            "wc_eqv",  # water/cement ratio (equivalent)
            "CEM_I_42.5_R",  # k=0
            "CEM_I_42.5_R+FA",  # k=0.5
            "CEM_I_42.5_R+SF",  # k=2.0
            "CEM_III/B_42.5",
        ]
    )  # k=0
    df["wc_eqv"] = wc_eqv
    df["CEM_I_42.5_R"] = np.array([np.nan, 3.1, 5.2, 6.8, 9.8, 13.4])
    df["CEM_I_42.5_R+FA"] = np.array([np.nan, 0.3, 1.9, 2.4, 6.5, 8.3])
    df["CEM_I_42.5_R+SF"] = np.array([3.5, 5.5, np.nan, np.nan, 16.5, np.nan])
    df["CEM_III/B_42.5"] = np.array([np.nan, 8.3, 16.9, 26.6, 44.3, 80.0])
    df = df.set_index("wc_eqv")
    return df


def k_e(pars):
    """ Calcualte k_e[-], envrionmental factor, effect of relative humidity

    Parameters
    ----------
    pars.RH_ref : 65 [%]
    g_e    : 2.5 [-]
    f_e    : 5.0 [-]
    """
    RH_real = pars.RH_real
    RH_ref = 65.0
    g_e = 2.5
    f_e = 5.0
    k_e = ((1 - (RH_real / 100) ** f_e) / (1 - (RH_ref / 100) ** f_e)) ** g_e
    return k_e


def k_c(pars):
    """ calculate k_c: execution transfer parameter [-], effect of period of curing for the accelerated carbonation test

    Parameters
    ----------
    pars.t_c: period of curing [d]
         constant
    b_c: exponent of regression [-]
         normal distribution, m: -0.567
                              s: 0.024
    """
    t_c = pars.t_c
    b_c = hf.Normal_custom(m=-0.567, s=0.024)
    k_c = (t_c / 7.0) ** b_c
    return k_c


def R_ACC_0_inv(pars):
    """ Calculate R_ACC_0_inv[(mm^2/year)/(kg/m^3)], the inverse effective carbonation resistance of concrete(accelerated)
        From ACC test or from existion empirical data interpolation for orientation purpose
        test condition: duration time = 56 days CO2 = 2.0 vol%, T =25 degC RH_ref =65

    Parameters
    ----------
    pars.x_c : float
                measured carbonation depth in the accelerated test[m]
    pars.option.choose : bool
                if true -> choose to use interpolation method
    pars.option.df_R_ACC : pd.dataframe
                data table for interpolate, loaded by function load_df_R_ACC, interpolated by function interp_extrap_f

    Returns
    -------
    out: numpy arrays
        parameter value with sample number = N_SAMPLE(defined globally)

    Notes
    -----
    Pay special attention to the units in the source code
    """
    x_c = pars.x_c
    if isinstance(x_c, int) or isinstance(x_c, float):
        # though acc-test
        tau = 420.0  # tau: 'time constant' in [(s/kg/m^3)^0.5], for described test conditions tau = 420
        R_ACC_0_inv_mean = (x_c / tau) ** 2  # [(m^2/s)/(kg/m^3)]

        # R_ACC_0_inv[10^-11*(m^2/s)/(kg/m^3)] ND(s = 0.69*m**0.78)
        R_ACC_0_inv_stdev = (
            1e-11 * 0.69 * (R_ACC_0_inv_mean * 1e11) ** 0.78
        )  # [(m^2/s)/(kg/m^3)]

        R_ACC_0_inv_temp = hf.Normal_custom(
            R_ACC_0_inv_mean, R_ACC_0_inv_stdev
        )  # [(m^2/s)/(kg/m^3)]

    elif pars.option.choose:
        #  'No test data, interpolate: orientation purpose'
        logger.warning("No test data, interpolate: orientation purpose")
        df = pars.option.df_R_ACC
        fit_df = df[pars.option.cement_type].dropna()

        # Curve fit
        x = fit_df.index.astype(float).values
        y = fit_df.values
        R_ACC_0_inv_mean = (
            hf.interp_extrap_f(x, y, pars.option.wc_eqv, plot=False) * 1e-11
        )  # [(m^2/s)/(kg/m^3)] #interp_extrap_f: defined function

        # R_ACC_0_inv[10^-11*(m^2/s)/(kg/m^3)] ND(s = 0.69*m**0.78)
        R_ACC_0_inv_stdev = (
            1e-11 * 0.69 * (R_ACC_0_inv_mean * 1e11) ** 0.78
        )  # [(m^2/s)/(kg/m^3)]

        R_ACC_0_inv_temp = hf.Normal_custom(
            R_ACC_0_inv_mean, R_ACC_0_inv_stdev
        )  # [(m^2/s)/(kg/m^3)]

    else:
        logger.error("R_ACC_0_inv calculation failed; application interupted")

        sys.exit("Error message")
    # unit change [(m^2/s)/(kg/m^3)] -> [(mm^2/year)/(kg/m^3)]  final model input
    R_ACC_0_inv_final = 365 * 24 * 3600 * 1e6 * R_ACC_0_inv_temp
    return R_ACC_0_inv_final


# Test method factors
def k_t():
    """Calculate test method regression parameter k_t[-]

    Notes
    -----
    for R_ACC_0_inv[(mm^2/years)/(kg/m^3)]"""
    k_t = hf.Normal_custom(1.25, 0.35)
    return k_t


def eps_t():
    """Calculate error term, eps_t[(mm^2/years)/(kg/m^3)],
    considering inaccuracies which occur conditionally when using the ACC test method  k_t[-]

    Notes
    -----
    for R_ACC_0_inv[(mm^2/years)/(kg/m^3)]"""
    eps_t = hf.Normal_custom(315.5, 48)
    return eps_t


# Evnironmental impact C_S
def C_S(C_S_emi=0):
    """Calculate CO2 density[kg/m^3] in the environment, it is about 350-380 ppm in the atm plus other source or sink

    Parameters
    ----------
    C_S_emi : additional emission, positive or negative(sink), default is 0
    """
    C_S_atm = hf.Normal_custom(0.00082, 0.0001)
    C_S = C_S_atm + C_S_emi
    return C_S


# weather function
def W_t(t, pars):
    """ Calcuate weather function W_t, a parameter considering the meso-climatic conditions due to wetting events of concrete surface

    Parameters
    ----------
    pars.ToW : time of wetness [-]
               ToW = (days with rainfall h_Nd >= 2.5 mm per day)/365

    pars.p_SR : probability of driving rain [-]
                Vertical -> weather station
                Horizontal 1.0
                Interior 0.0

    pars.b_w; exponent of regression [-] ND(0.446, 0.163)
    built-in param t_0 : time of reference [years]

    Returns
    -------
    out : numpy array
    """
    ToW = pars.ToW
    p_SR = pars.p_SR

    t_0 = 0.0767  # [year]
    b_w = hf.Normal_custom(0.446, 0.163)

    W = (t_0 / t) ** ((p_SR * ToW) ** b_w / 2.0)
    return W


# helper function: calibration fucntion
def calibrate_f(model_raw, t, carb_depth_field, tol=1e-6, max_count=50, print_out=True):
    """carb_depth_field[mm]-> find corresponding x_c(accelerated test carb depth[m])
    Calibrate the carbonation model with field carbonation test data and return the new calibrated model object/instance
    Optimization metheod: searching for the best accelerated test carbonation depth x_c[m] so the model matches field data
    on the mean value of the carbonation depth)

    Parameters
    ----------
    model_raw : object/instance of Carbonation_Model class, mutable, so a deepcopy will be used in this function
    t         : float or int
                survey time, age of the concrete[year]
    carb_depth_field : numpy array
                       at time t, field carbonation depths[mm]
    tol : float
        accelerated carbonation depth x_c optimization tolerance, default is 1e-5 [mm]
    max_count : int
                maximun number of searching iteration, default is 50

    Returns
    -------
    out : object/instance of Carbonation_Model class
         new calibrated model
    """
    model = model_raw.copy()
    # accelrated test
    # cap
    x_c_min = 0.0
    x_c_max = 0.1  # [m] unrealisticall large safe ceiling

    # optimization
    count = 0
    while x_c_max - x_c_min > tol:
        # update guess
        x_c_guess = 0.5 * (x_c_min + x_c_max)
        model.pars.x_c = x_c_guess
        model.run(t)
        carb_depth_mean = hf.Get_mean(model.xc_t)

        # compare
        if carb_depth_mean < carb_depth_field.mean():
            # narrow the cap
            x_c_min = max(x_c_guess, x_c_min)
        else:
            x_c_max = min(x_c_guess, x_c_max)

        logger.info("carb_depth_mean:{}".format(carb_depth_mean))
        logger.info("x_c:{}".format(x_c_guess))
        logger.debug("cap:[{}{}]".format(x_c_min, x_c_max))

        count += 1
        if count > max_count:
            logger.warning("iteration exceeded max {}".format(count))
            break

    if print_out:
        print("carb_depth:")
        print(
            "model: \nmean:{}\nstd:{}".format(
                hf.Get_mean(model.xc_t), hf.Get_std(model.xc_t)
            )
        )
        print(
            "field: \nmean:{}\nstd:{}".format(
                carb_depth_field.mean(), carb_depth_field.std()
            )
        )
    return model


def carb_year(model, year_lis, plot=True, amplify=80):
    """run model over time"""
    t_lis = year_lis
    M_cal = model

    M_lis = []
    for t in t_lis:
        M_cal.run(t)
        M_cal.postproc()
        M_lis.append(M_cal.copy())
    if plot:
        fig, [ax1, ax2, ax3] = plt.subplots(
            nrows=3,
            figsize=(8, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [1, 1, 3]},
        )
        # plot a few distrubtion
        indx = np.linspace(0, len(year_lis) - 1, min(6, len(year_lis))).astype("int")[
            1:
        ]
        M_sel = [M_lis[i] for i in indx]

        ax1.plot([this_M.t for this_M in M_lis], [this_M.pf for this_M in M_lis], "k--")
        ax1.plot(
            [this_M.t for this_M in M_sel],
            [this_M.pf for this_M in M_sel],
            "k|",
            markersize=15,
        )
        ax1.set_ylabel("Probability of failure $P_f$")

        ax2.plot(
            [this_M.t for this_M in M_lis],
            [this_M.beta_factor for this_M in M_lis],
            "k--",
        )
        ax2.plot(
            [this_M.t for this_M in M_sel],
            [this_M.beta_factor for this_M in M_sel],
            "k|",
            markersize=15,
        )
        ax2.set_ylabel(r"Reliability factor $\beta$")

        # plot mean results
        ax3.plot(t_lis, [M.pars.cover_mean for M in M_lis], "--C0")
        ax3.plot(t_lis, [hf.Get_mean(M.xc_t) for M in M_lis], "--C1")
        # plot distribution
        for this_M in M_sel:
            hf.RS_plot(this_M, ax=ax3, t_offset=this_M.t, amplify=amplify)

        import matplotlib.patches as mpatches

        R_patch = mpatches.Patch(color="C0", label="R: cover", alpha=0.8)
        S_patch = mpatches.Patch(color="C1", label="S: carbonation", alpha=0.8)

        ax3.set_xlabel("Time[year]")
        ax3.set_ylabel("cover/carbonation depth [mm]")
        ax3.legend(handles=[R_patch, S_patch], loc="upper left")

        plt.tight_layout()

    return [this_M.pf for this_M in M_lis], [this_M.beta_factor for this_M in M_lis]


class Carbonation_Model:
    def __init__(self, pars):
        self.pars = pars  # pars with user-input, then updated with derived parameters
        logger.debug("\nRaw pars are {}\n".format(vars(pars)))

    def run(self, t):
        """t[year]"""
        self.xc_t = Carb_depth(t, self.pars)
        self.t = t
        logger.info("Carbonation depth, xc_t{} mm".format(self.xc_t))

    def postproc(self, plot=False):
        sol = hf.Pf_RS(
            (self.pars.cover_mean, self.pars.cover_std), self.xc_t, plot=plot
        )
        self.pf = sol[0]
        self.beta_factor = sol[1]
        self.R_distrib = sol[2]
        self.S_kde_fit = sol[3]
        self.S = self.xc_t
        logger.info("pf{}\n beta_factor{}".format(self.pf, self.beta_factor))

    def calibrate(self, t, carb_depth_field, print_out=False):
        """return a new model instance with calibrated param"""
        model_cal = calibrate_f(self, t, carb_depth_field, print_out=print_out)
        return model_cal

    def copy(self):
        return deepcopy(self)

    def carb_with_year(self, year_lis, plot=True, amplify=80):
        pf_lis, beta_lis = carb_year(self, year_lis, plot=plot, amplify=amplify)
        return np.array(pf_lis), np.array(beta_lis)
