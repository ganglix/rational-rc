"""
**Summary**

Modified analytical solution of Fick’s law (square root of time)\n
Proportional constant is modified by material properties and exposure environments

+ **Resistance**: 	cover depth

+ **Load**: 		carbonation depth

+ **limit-state**: 	carbonation depth >= cover depth

+ **Field data**: 	carbonation depths (repeated measurements)

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from copy import deepcopy
import logging

import math_helper as mh

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
def carb_depth(t, pars):
    """ Calculate carbonation depth at a given time based on the parameters. 
    The derived parameters (including the k constant of sqrt of time) are also calculated within this function.

    Caution: The pars instance is mutable,so a deepcopy of the original instance should be used if the calculation is not intended for "inplace".

    Parameters
    ----------
    t      : time [year]
    pars   : object/instance of wrapper class (empty class)
               a wrapper of all material and environmental parameters deep-copied from the raw data

    Returns
    -------
    xc_t : carbonation depth at time t [mm]

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
    
    # Calculate intermediate parameters
    pars.t = t
    pars.k_e = k_e(pars)
    pars.k_c = k_c(pars)
    pars.k_t = k_t()
    pars.R_ACC_0_inv = R_ACC_0_inv(pars)
    pars.eps_t = eps_t()
    pars.C_S = C_S(C_S_emi=pars.C_S_emi)
    pars.W_t = W_t(t, pars)
    
    # Calculate carbonation depth
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
    pandas.DataFrame
        Dataframe containing the accelerated carbonation test data.

    Notes
    -----
    w/c 0.45 cemI is comparable to ACC of 3 mm.
    """
    wc_eqv = np.arange(0.35, 0.60 + (0.05 / 2), 0.05)

    data = {
        "wc_eqv": wc_eqv,
        "CEM_I_42.5_R": [np.nan, 3.1, 5.2, 6.8, 9.8, 13.4],
        "CEM_I_42.5_R+FA": [np.nan, 0.3, 1.9, 2.4, 6.5, 8.3],
        "CEM_I_42.5_R+SF": [3.5, 5.5, np.nan, np.nan, 16.5, np.nan],
        "CEM_III/B_42.5": [np.nan, 8.3, 16.9, 26.6, 44.3, 80.0]
    }

    df = pd.DataFrame(data)
    df.set_index("wc_eqv", inplace=True)

    return df

def k_e(pars):
    """ Calculate k_e[-], environmental factor, effect of relative humidity

    Parameters
    ----------
    pars.RH_ref : float
        Reference relative humidity 65 [%]
    g_e    : 2.5 [-]
    f_e    : 5.0 [-]

    Returns
    -------
    float
        Calculated environmental factor k_e[-]
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
    pars.t_c : float
        Period of curing [d]
    b_c: [built-in] exponent of regression [-]
         normal distribution, m: -0.567
                              s: 0.024
    Returns
    -------
    float
        Calculated execution transfer parameter k_c[-]
    """
    t_c = pars.t_c
    b_c = mh.normal_custom(m=-0.567, s=0.024)
    k_c = (t_c / 7.0) ** b_c
    return k_c


def R_ACC_0_inv(pars):
    """ Calculate R_ACC_0_inv[(mm^2/year)/(kg/m^3)], the inverse effective carbonation resistance of concrete(accelerated)
    From ACC test or from existing empirical data interpolation for orientation purpose
    test condition: duration time = 56 days CO2 = 2.0 vol%, T =25 degC RH_ref =65

    Parameters
    ----------
    pars.x_c : float
        Measured carbonation depth in the accelerated test [m]
    pars.option.choose : bool
        If True, choose to use interpolation method
    pars.option.df_R_ACC : pandas.DataFrame
        Data table for interpolation, loaded by function load_df_R_ACC, interpolated by function interp_extrap_f

    Returns
    -------
    out: numpy arrays
        Calculated inverse effective carbonation resistance [mm^2/year)/(kg/m^3] 
        with sample number = N_SAMPLE (defined globally)

    Notes
    -----
    Pay special attention to the units in the source code
    """
    x_c = pars.x_c
    if isinstance(x_c, int) or isinstance(x_c, float):
        # Through acc-test
        tau = 420.0  # tau: 'time constant' in [(s/kg/m^3)^0.5], for described test conditions tau = 420
        R_ACC_0_inv_mean = (x_c / tau) ** 2  # [(m^2/s)/(kg/m^3)]

        # R_ACC_0_inv[10^-11*(m^2/s)/(kg/m^3)] ND(s = 0.69*m**0.78)
        R_ACC_0_inv_stdev = (
            1e-11 * 0.69 * (R_ACC_0_inv_mean * 1e11) ** 0.78
        )  # [(m^2/s)/(kg/m^3)]

        R_ACC_0_inv_temp = mh.normal_custom(
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
            mh.interp_extrap_f(x, y, pars.option.wc_eqv, plot=False) * 1e-11
        )  # [(m^2/s)/(kg/m^3)] #interp_extrap_f: defined function

        # R_ACC_0_inv[10^-11*(m^2/s)/(kg/m^3)] ND(s = 0.69*m**0.78)
        R_ACC_0_inv_stdev = (
            1e-11 * 0.69 * (R_ACC_0_inv_mean * 1e11) ** 0.78
        )  # [(m^2/s)/(kg/m^3)]

        R_ACC_0_inv_temp = mh.normal_custom(
            R_ACC_0_inv_mean, R_ACC_0_inv_stdev
        )  # [(m^2/s)/(kg/m^3)]

    else:
        logger.error("R_ACC_0_inv calculation failed; application interrupted")
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
    k_t = mh.normal_custom(1.25, 0.35)
    return k_t


def eps_t():
    """Calculate error term, eps_t[(mm^2/years)/(kg/m^3)],
    considering inaccuracies which occur conditionally when using the ACC test method  k_t[-]

    Notes
    -----
    for R_ACC_0_inv[(mm^2/years)/(kg/m^3)]"""
    eps_t = mh.normal_custom(315.5, 48)
    return eps_t


# Environmental impact C_S
def C_S(C_S_emi=0):
    """Calculate CO2 density[kg/m^3] in the environment; it is about 350-380 ppm in the atm plus other source or sink

    Parameters
    ----------
    C_S_emi : additional emission, positive or negative(sink), default is 0

    Returns
    -------
    float
        CO2 density in kg/m^3
    """
    C_S_atm = mh.normal_custom(0.00082, 0.0001)
    C_S = C_S_atm + C_S_emi
    return C_S


# weather function
def W_t(t, pars):
    """ Calculate weather parameter W, a parameter considering the meso-climatic conditions due to wetting events of concrete surface
    
    Parameters
    ----------
    t : float
        Time [years]
    pars : object
        Instance of Param class containing the following attributes:
        - ToW : float
            Time of wetness [-], calculated as (days with rainfall h_Nd >= 2.5 mm per day)/365
        - p_SR : float
            Probability of driving rain [-], 1.0 for vertical surface, 0.0 for horizontal or interior surfaces
        - b_w : float
            Exponent of regression [-], normally distributed with mean=0.446 and standard deviation=0.163
        - t_0 : float
            Time of reference [years] [built-in param]

    Returns
    -------
    numpy array
        Weather parameter array W
    """

    ToW = pars.ToW
    p_SR = pars.p_SR

    t_0 = 0.0767  # [year]
    b_w = mh.normal_custom(0.446, 0.163)

    W = (t_0 / t) ** ((p_SR * ToW) ** b_w / 2.0)
    return W


# helper function: calibration function
def calibrate_f(model_raw, t, carb_depth_field, tol=1e-6, max_count=50, print_out=True):
    """Calibrate the carbonation model with field carbonation test data [mm] and return the new calibrated model.
    Optimization method: searching for the best accelerated test carbonation depth x_c[m] so the model matches field data
    on the mean value of the carbonation depth)

    Parameters
    ----------
    model_raw : object
        Instance of the CarbonationModel class, mutable (a deepcopy will be used in this function).
    t : float or int
        Survey time, age of the concrete [years].
    carb_depth_field : numpy array
        Field carbonation depths at time t [mm].
    tol : float, optional
        Accelerated carbonation depth x_c optimization tolerance, default is 1e-5 [mm].
    max_count : int, optional
        Maximum number of searching iterations, default is 50.
    print_out : bool, optional
        Flag to print out the results, default is True.

    Returns
    -------
    object
        New calibrated model instance.
    """

    model = model_raw.copy()

    # Define the initial search space for x_c
    x_c_min = 0.0
    x_c_max = 0.1  # [m] unrealistically large safe ceiling

    # Optimization
    count = 0
    while x_c_max - x_c_min > tol:
        # update guess for x_c
        x_c_guess = 0.5 * (x_c_min + x_c_max)
        model.pars.x_c = x_c_guess
        model.run(t)
        carb_depth_mean = mh.get_mean(model.xc_t)

        # Compare the mean carbonation depth with the field data
        if carb_depth_mean < carb_depth_field.mean():
            # Narrow the search space
            x_c_min = max(x_c_guess, x_c_min)
        else:
            x_c_max = min(x_c_guess, x_c_max)

        logger.info("carb_depth_mean:{}".format(carb_depth_mean))
        logger.info("x_c:{}".format(x_c_guess))
        logger.debug("cap:[{}{}]".format(x_c_min, x_c_max))

        count += 1
        if count > max_count:
            logger.warning("Iteration exceeded the maximum count {}".format(count))
            break

    if print_out:
        print("carb_depth:")
        print(
            "model: \nmean:{}\nstd:{}".format(
                mh.get_mean(model.xc_t), mh.get_std(model.xc_t)
            )
        )
        print(
            "field: \nmean:{}\nstd:{}".format(
                carb_depth_field.mean(), carb_depth_field.std()
            )
        )
    return model


def carb_year(model, year_lis, plot=True, amplify=80):
    """Run the model over time and plot the results."""
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

        # plot a few distributions
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

        # Plot mean results
        ax3.plot(t_lis, [M.pars.cover_mean for M in M_lis], "--C0")
        ax3.plot(t_lis, [mh.get_mean(M.xc_t) for M in M_lis], "--C1")

        # Plot distribution
        for this_M in M_sel:
            mh.plot_RS(this_M, ax=ax3, t_offset=this_M.t, amplify=amplify)

        import matplotlib.patches as mpatches

        R_patch = mpatches.Patch(color="C0", label="R: cover", alpha=0.8)
        S_patch = mpatches.Patch(color="C1", label="S: carbonation", alpha=0.8)

        ax3.set_xlabel("Time[year]")
        ax3.set_ylabel("Cover/Carbonation Depth [mm]")
        ax3.legend(handles=[R_patch, S_patch], loc="upper left")

        plt.tight_layout()

    return [this_M.pf for this_M in M_lis], [this_M.beta_factor for this_M in M_lis]


class CarbonationModel:
    def __init__(self, pars):
        self.pars = pars  # pars with user-input, then updated with derived parameters
        logger.debug("\nRaw pars are {}\n".format(vars(pars)))

    def run(self, t):
        """Run the model for the given time t [year]."""
        self.xc_t = carb_depth(t, self.pars)
        self.t = t
        logger.info("Carbonation depth, xc_t: {} mm".format(self.xc_t))

    def postproc(self, plot=False):
        """Post-process the model results."""
        sol = mh.pf_RS(
            (self.pars.cover_mean, self.pars.cover_std), self.xc_t, plot=plot
        )
        self.pf = sol[0]
        self.beta_factor = sol[1]
        self.R_distrib = sol[2]
        self.S_kde_fit = sol[3]
        self.S = self.xc_t
        logger.info("pf{}\n beta_factor{}".format(self.pf, self.beta_factor))

    def calibrate(self, t, carb_depth_field, print_out=False):
        """Calibrate the model with field data and return a new calibrated model instance."""
        model_cal = calibrate_f(self, t, carb_depth_field, print_out=print_out)
        return model_cal

    def copy(self):
        """Create a deep copy of the model."""
        return deepcopy(self)

    def carb_with_year(self, year_lis, plot=True, amplify=80):
        """Run the model over time and return lists of pf and beta values."""
        pf_lis, beta_lis = carb_year(self, year_lis, plot=plot, amplify=amplify)
        return np.array(pf_lis), np.array(beta_lis)
