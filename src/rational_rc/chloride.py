"""
**Summary**

Analytical solution of Fick’s second law under the advection zone,\n
modified with material property and exposure environment.

+ **Resistance**: 	critical chloride content
+ **Load**: 		chloride content at rebar depth
+ **limit-state**: 	chloride content at rebar depth >= critical chloride content
+ **Field data**: 	chloride content profile

Future TODO: make t input vectorized. 
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.special import erf

import matplotlib.pyplot as plt
import sys
from copy import deepcopy
import logging

import rational_rc.math_helper as mh


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
def chloride_content(x, t, pars):
    """Calculate the chloride content at depth x and time t with Fick's 2nd law below the advection zone (x > dx).
    
    + Caution: The pars instance is mutable, so a deepcopy of the original instance should be used if the calculation is not intended for "inplace".

    Parameters
    ----------
    x : float or int
        Depth at which chloride content C_x_t is reported [mm].
    t : float or int
        Time [year].
    pars : param object
        An instance of the Param class, containing material and environmental parameters.

    Returns
    -------
    numpy array
        Sample of the distribution of the chloride content in concrete at a depth x (surface x=0) at time t [wt-.%/c]
    
    Note
    ----
    Intermediate parameters are calculated and attached to pars:

    C_0    : initial chloride content of the concrete [wt-.%/cement]
    C_S_dx : chloride content at a depth dx and a certain point of time t [wt-.%/cement]
    dx     : depth of the advection zone (concrete layer, up to which the process of chloride penetration differs from Fick’s 2nd law of diffusion) [mm]
    D_app  : apparent coefficient of chloride diffusion through concrete [mm^2/year]
    erf    : imported error function
    """
    pars.D_app = D_app(t, pars)
    C_x_t = pars.C_0 + (pars.C_S_dx - pars.C_0) * (
        1 - erf((x - pars.dx) / (2 * (pars.D_app * t) ** 0.5))
    )
    return C_x_t


def D_app(t, pars):
    """ Calculate the apparent coefficient of chloride diffusion through concrete D_app[mm^2/year]

    Parameters
    ----------
    t : float, int
        time [year]
    pars : instance of param object
        a wrapper of all material and environmental parameters deep-copied from the raw data

    Returns
    -------
    numpy array
        sample of the distribution of the apparent coefficient of chloride diffusion through concrete [mm^2/year]

    Note
    ----
    intermediate parameters calculated and attached to pars
    
    k_e     : environmental transfer variable [-]
    D_RCM_0 : chloride migration coefficient [mm^2/year]
    k_t     : transfer parameter, k_t =1 was set in A_t()[-]
    A_t     : subfunction considering the 'ageing' [-]
    """
    pars.k_e = k_e(pars)
    pars.D_RCM_0 = D_RCM_0(pars)

    pars.A_t = A_t(t, pars)  # pars.k_t =1 was set in A_t()
    D_app = pars.k_e * pars.D_RCM_0 * pars.k_t * pars.A_t
    return D_app


def k_e(pars):
    """
    Calculate the environmental transfer variable k_e [-].

    Parameters
    ----------
    pars : Param
        An instance of the Param class, containing the following parameters:
    pars.T_ref : float
        Standard test temperature 293.[K].
    pars.T_real : float
        Temperature of the structural element [K].
    pars.b_e : float
        Regression variable [K].

    Returns
    -------
    numpy array
        Large sample of the distribution of k_e.
    """
    pars.T_ref = 293  # K (20°C)
    pars.b_e = b_e()
    k_e = np.e ** (pars.b_e * (1 / pars.T_ref - 1 / pars.T_real))
    return k_e


def b_e():
    """Provide a large sample array of b_e: regression variable [K]."""
    b_e = mh.normal_custom(4800, 700)  # K
    return b_e


def A_t(t, pars):
    """Calculate A_t considering the ageing effect.

    Parameters
    ----------
    t : int or float
        Time [year].
    pars : instance of param object
        A wrapper of all material and environmental parameters deep-copied from the raw data.
        
        pars.concrete_type : string
            Option:

            'Portland cement concrete',

            'Portland fly ash cement concrete',

            'Blast furnace slag cement concrete'

    Returns
    -------
    numpy array
        Subfunction considering the "ageing" [-].

    Note
    ----
    Built-in parameters
        pars.k_t : transfer parameter, k_t =1 was set for experiment [-]
        pars.t_0 : reference point of time, 0.0767 [year]
    """
    pars.t_0 = 0.0767  # reference point of time [year]
    # To carry out the quantification of a, the transfer variable k_t was set to k_t = 1:
    pars.k_t = 1
    # a: aging exponent
    a = None
    if pars.concrete_type == "Portland cement concrete":
        # CEM I; 0.40 ≤ w/c ≤ 0.60
        a = mh.beta_custom(0.3, 0.12, 0.0, 1.0)

    elif pars.concrete_type == "Portland fly ash cement concrete":
        # f≥0.20·z;k=0.50; 0.40≤w/c_eqv. ≤0.62
        a = mh.beta_custom(0.6, 0.15, 0.0, 1.0)

    elif pars.concrete_type == "Blast furnace slag cement concrete":
        # CEM III/B; 0.40 ≤ w/c ≤ 0.60
        a = mh.beta_custom(0.45, 0.20, 0.0, 1.0)

    A = (pars.t_0 / t) ** a
    return A


def D_RCM_0(pars):
    """ Return the chloride migration coefficient from Rapid chloride migration test [m^2/s] see NT Build 492
    if the test data is not available from pars, use interpolation of existing empirical data for orientation purpose
    Pay attention to the units output [mm^2/year], used for the model

    Parameters
    ----------
    pars : instance of param object
        a wrapper of all material and environmental parameters deep-copied from the raw data

    pars.D_RCM_test    : int or float
                            RCM test results[m^2/s], the mean value from the test is used, and standard deviation is estimated based on mean
    pars.option.choose : bool
                            If true interpolation from existing data table is used
    pars.option.df_D_RCM_0  : pandas.DataFrame
                            Experimental data table(cement type, and w/c eqv) for interpolation
    pars.option.cement_type : string
                            Select cement type for data interpolation of the df_D_RCM_0,
                            Options:
                            'CEM_I_42.5_R'\n
                            'CEM_I_42.5_R+FA'\n
                            'CEM_I_42.5_R+SF'\n
                            'CEM_III/B_42.5'
    pars.option.wc_eqv : float
                        equivalent water cement ratio considering supplementary cementitious materials

    Returns
    -------
    numpy array
         D_RCM_0_final [mm^2/year].
    """
    if isinstance(pars.D_RCM_test, int) or isinstance(pars.D_RCM_test, float):
        # though test result [m^2/s]
        D_RCM_0_mean = pars.D_RCM_test  # [m^2/s]
        D_RCM_0_std = 0.2 * D_RCM_0_mean
        D_RCM_0_temp = mh.normal_custom(D_RCM_0_mean, D_RCM_0_std)  # [m^2/s]
    elif pars.option.choose:
        # print 'No test data, interpolate: orientation purpose'
        df = pars.option.df_D_RCM_0
        fit_df = df[pars.option.cement_type].dropna()

        # Curve fit
        x = fit_df.index.astype(float).values
        y = fit_df.values
        # [m^2/s] #interp_extrap_f: defined function
        D_RCM_0_mean = mh.interp_extrap_f(x, y, pars.option.wc_eqv, plot=False) * 1e-12
        D_RCM_0_std = 0.2 * D_RCM_0_mean  # [m^2/s]

        D_RCM_0_temp = mh.normal_custom(D_RCM_0_mean, D_RCM_0_std)  # [m^2/s]

    else:
        print("D_RCM_0 calculation failed.")
        sys.exit("Error message")

    # unit change [m^2/s] -> [mm^2/year]  final model input
    D_RCM_0_final = 1e6 * 3600 * 24 * 365 * D_RCM_0_temp
    return D_RCM_0_final


# Built-in Data Table for data interpolation

# Data table to interpolate/extrapolate
def load_df_D_RCM():
    """Load the data table of the Rapid Chloride Migration(RCM) test
    for D_RCM interpolation.

    Returns
    -------
    pandas.DataFrame
        Data table from the RCM experiment.
    """
    wc_eqv = np.arange(0.35, 0.60 + (0.05 / 2), 0.05)

    data = {
        "wc_eqv": wc_eqv,
        "CEM_I_42.5_R": [np.nan, 8.9, 10.0, 15.8, 17.9, 25.0],
        "CEM_I_42.5_R+FA": [np.nan, 5.6, 6.9, 9.0, 10.9, 14.9],
        "CEM_I_42.5_R+SF": [4.4, 4.8, np.nan, np.nan, 5.3, np.nan],
        "CEM_III/B_42.5": [np.nan, 8.3, 1.9, 2.8, 3.0, 3.4]
    }

    df = pd.DataFrame(data)
    df = df.set_index("wc_eqv")
    return df


def C_eqv_to_C_S_0(C_eqv):
    """ Convert aqueous solution chloride content to saturated chloride content in concrete
    interpolate function for 300kg cement w/c=0.5 OPC. Other empirical function should be used if available

    Parameters
    ----------
    C_eqv : float
            chloride content of the solution at the surface[g/L]

    Returns
    -------
    float
        saturated chloride content in concrete[wt-%/cement]
    """
    #  chloride content of the solution at the surface[g/L]
    x = np.array([0.0, 0.25, 0.93, 2.62, 6.14, 9.12, 13.10, 20.18, 25.03, 30.0])
    # saturated chloride content in concrete[wt-%/cement]
    y = np.array([0.0, 0.26, 0.47, 0.74, 1.13, 1.39, 1.70, 2.19, 2.49, 2.78])

    f = interp1d(x, y)
    if C_eqv <= x.max():
        C_S_0 = f(C_eqv)
    else:
        print("warning: C_eqv_to_C_S_0 extrapolation used!")
        C_S_0 = mh.interp_extrap_f(x[-5:-1], y[-5:-1], C_eqv, plot=False)
    return C_S_0


# C_S: chloride content at surface = C_S_dx when dx = 0
# C_S_dx: chloride content at subsurface

# Environmental param: Potential chloride impact C_eqv
def C_eqv(pars):
    """Calculate equivalent chloride solution concentration, C_eqv[g/L] as a measurement of the Potential chloride impact
    from the source of 

    1. marine or coastal and/or 
    2. de icing salt 

    It is later used to estimate the boundary condition C_S_dx of contineous exposure or non-geometry-sensitive intermittent exposure
    
    Parameters
    ----------
    pars : instance of the param object
        a wrapper of all material and environmental parameters deep-copied from the raw data
        See Note for details


    Returns
    -------
    float
          C_eqv, equivalent chloride solution concentration [g/L]

    Note
    ----
    1. marine or coastal

    pars.C_0_M : natural chloride content of sea water [g/l]

    2. de-icing salt

    pars.C_0_R : average chloride content of the chloride contaminated water [g/l]
    pars.n     : average number of salting events per year [-]
    pars.C_R_i : average amount of chloride spread within one spreading event [g/m2]
    pars.h_S_i : amount of water from rain and melted snow per spreading period [l/m2]

    C_eqv is used for continuous exposure or non-geometry-sensitive intermittent exposure.
    For geometry-sensitive conditions (roadside splash), the tested C_max() should be used.
    """
    C_0_M = pars.C_0_M
    C_0_R = (pars.n * pars.C_R_i) / pars.h_S_i

    if pars.marine:
        C_eqv = C_0_M + C_0_R
    else:
        C_eqv = C_0_R

    return C_eqv


# exposure condition
def C_S_0(pars):
    """Return (surface) chloride saturation concentration C_S_0 [wt.-%/cement] caused by  C_eqv [g/l].

    Parameters
    ----------
    pars.C_eqv : float
                 calculated with by C_eqv(pars) [g/L]
    pars.C_eqv_to_C_S_0 : global function
        This function is based experiment with the info of

        binder-specific chloride-adsorption-isotherms
        the concrete composition(cement/concrete ratio)
        potential chloride impact C_eqv [g/L]
    
    Returns
    -------
    float
        chloride saturation concentration C_S_0 [wt.-%/cement]

    Note
    ----
    The conversion function C_eqv_to_C_S_0(pars.C_eqv) is derived from experiment data of 300kg cement w/c=0.5 OPC. 
    TODO: update to a conversion function dependent on the proportioning and cementitious material
    """
    # -> get the relationship
    pars.C_eqv = C_eqv(pars)
    C_S_0 = pars.C_eqv_to_C_S_0(pars.C_eqv)
    return C_S_0


# substitute chloride surface concentration
def C_S_dx(pars):
    """Return the substitute chloride surface concentration, i.e. chloride content just below the advection zone.
    
    Parameters
    ----------
    pars       : object/instance of param class
                 contains material and environment parameters

    pars.C_S_0 : float or numpy array
                chloride saturation concentration C_S_0 [wt.-%/cement]
                built-in calculation with C_S_0(pars)
    pars.C_max : float
                maximum content of chlorides within the chloride profile, [wt.-%/cement]
                built-in calculation with C_max(pars)
    pars.exposure_condition : string
                    continuous/intermittent exposure - 'submerged','leakage', 'spray', 'splash'

    pars.exposure_condition_geom_sensitive : bool
                if True, the C_max is used instead of C_S_0

    Returns
    -------
    float or numpy arrays
          C_S_dx, the substitute chloride surface concentration [wt.-%/cement]

    Note
    ----
    Fick's 2nd law applies below the advection zone (depth=dx). No advection effect when dx = 0.
    Conditions considered: continuous/intermittent exposure - 'submerged', 'leakage', 'spray', 'splash' where C_S_dx = C_S_0.
    The advection depth dx is calculated in the dx() function externally.

    If exposure_condition_geom_sensitive is True, the observed/empirical highest chloride content in concrete C_max is used.
    C_max is calculated by C_max().
    """

    pars.C_S_0 = C_S_0(pars)
    # transfer functions considering geometry and exposure conditions
    # C_S_dx considered as time independent for simplification
    if pars.exposure_condition in ["submerged", "leakage", "spray"]:
        # for continuous exposure, such as submerge: use transfer function dx=0
        C_S_dx = pars.C_S_0  # dx = 0, set in dx()

    elif pars.exposure_condition == "splash":
        if pars.exposure_condition_geom_sensitive:
            # geometry-sensitive road splash use C_max
            pars.C_max = C_max(pars)
            C_S_dx_mean = pars.C_max
            C_S_dx_std = 0.75 * C_S_dx_mean
            C_S_dx = mh.normal_custom(C_S_dx_mean, C_S_dx_std, non_negative=True)
        else:
            # intermittent exposure, dx >0, set in dx()
            C_S_dx = pars.C_S_0
    else:
        C_S_dx = None
        logger.warning("C_S_dx calculation failed")
    return C_S_dx


# Convection depth
def dx(pars):
    """Return dx: advection depth [mm] dependent on the exposure conditions.

    Parameters
    ----------
    pars : object/instance of param class
        Contains material and environment parameters.

    pars.exposure_condition : string
        Exposure condition - 'splash', 'submerged', 'leakage', 'spray'.

    Returns
    -------
    float
        dx, advection depth [mm].

    Note
    ----
    The advection depth dx is specific to each exposure condition.
    """

    
    condition = pars.exposure_condition
    dx = None
    if condition == "splash":
        # - for splash conditions (splash road environment, splash marine environment)
        dx = mh.beta_custom(5.6, 8.9, 0.0, 50.0)

    if condition in ["submerged", "leakage", "spray"]:
        # - for submerged marine structures
        # - for leakage due to seawater and constant ground water level
        # - for spray conditions(spray road environment, spray marine environment)
        #   a height of more than 1.50 m above the road (spray zone) no dx develops
        dx = 0.0

    if condition == "other":
        print("To be determined")
        pass

    return dx


#  Chloride surface content CS resp. substitute chloride surface content C_S_dx
def C_max(pars):
    """
    Calculate C_max: maximum content of chlorides within the chloride profile [wt.-%/cement].
    It is calculated from empirical equations or from test data [wt.-%/concrete].

    Parameters
    ----------
    pars : object/instance of param class
    Contains material and environment parameters.

    pars.cement_concrete_ratio : float
                      cement/concrete weight ratio, used to convert [wt.-%/concrete] -> [wt.-%/cement]

    pars.C_max_option : string
        Option for determining C_max:
        - "empirical" - use empirical equation.
        - "user_input" - use user input from test.

    pars.x_a : float
        "empirical" option: horizontal distance from the roadside [cm]

    pars.x_h : float
        "empirical" option: height above road surface [cm]

    pars.C_max_user_input : float
        "user_input" option: Experiment-tested maximum chloride content [wt.-%/concrete]

    Returns
    -------
    float
        C_max, maximum content of chlorides within the chloride profile, [wt.-%/cement]

    Note
    ----
    The empirical expression should be determined for structures of different exposure or concrete mix.
    A typical C_max used by default in this function is from

    + location: urban and rural areas in Germany
    + time of exposure of the considered structure: 5-40 years
    + concrete: CEM I, w/c = 0.45 up to w/c = 0.60,

    """
    C_max_temp = None
    if pars.C_max_option == "empirical":
        # empirical eq should be determined for structures of different exposure or concrete mixes????????
        # A typical C_max
        # – location: urban and rural areas in Germany
        # – time of exposure of the considered structure: 5-40 years
        # – concrete: CEM I, w/c = 0.45 up to w/c = 0.60,
        x_a = pars.x_a
        x_h = pars.x_h
        C_max_temp = (
            0.465 - 0.051 * np.log(x_a + 1) - (0.00065 * (x_a + 1) ** -0.187) * x_h
        )  # wt.%/concrete

    if pars.C_max_option == "user_input":
        C_max_temp = pars.C_max_user_input  # wt-% concrete

    # wt.%/concrete -> wt.%cement
    C_max_final = C_max_temp / pars.cement_concrete_ratio
    return C_max_final


# critical chloride content
def C_crit_param():
    """Return the beta distribution parameters for the critical chloride content(total chloride), C_crit [wt.-%/cement]

    Returns
    -------
    tuple
         parameters of general beta distribution (mean, std, lower_bound, upper_bound)
    """
    C_crit_param = (0.6, 0.15, 0.2, 2.0)
    return C_crit_param


# calibration function
def calibrate_chloride_f(
    model_raw,
    x,
    t,
    chloride_content,
    tol=1e-15,
    max_count=50,
    print_out=True,
    print_proc=False,
):
    """
    Calibrate the chloride model with field chloride test data (at one depth and at one time) and return the new calibrated model object/instance

    Parameters
    ----------
    model_raw : object/instance of Chloride_model class (to be calibrated)
    x         : float
                depth [mm]
    t:        : int or float
                time [year]
    chloride_content : float or int
                       field chloride_content[wt.-%/cement] at time t, depth x,

    tol: float
         D_RCM_0 optimization absolute tolerance 1e-15 [m^2/s]

    max_count : int
                maximum number of searching iteration, default is 50
    print_out : bool
                if true, print model and field chloride content
    print_proc: bool
                if turn, print optimization process. (debug message in the logger)

    Returns
    -------
    instance of Chloride_Model object
          new calibrated model

    Note
    ----
    Optimization method:  Field chloride content at depth x and time t -> find corresponding D_RCM_test 
    and then update D_RCM_0(repaid chloride migration diffusivity[m^2/s]) in the model.run()

    calibrate model to field data at three depths in calibrate_chloride_f_group()
    chloride_content_field[wt.-%/cement] at time t

        + optimizing corresponding D_RCM_test,
        + fixed C_S_dx (exposure type dependent)
        + fixed dx (determined by the original model)
    """

    model = model_raw.copy()
    # target chloride content at depth x
    cl = chloride_content

    # DCM test
    # cap
    D_RCM_test_min = 0.0
    # [m/s] unrealistically large safe ceiling corresponding to a D_RCM_0= [94] [mm/year]
    D_RCM_test_max = 3e-12

    # optimization
    count = 0
    while D_RCM_test_max - D_RCM_test_min > tol:
        # update guess
        D_RCM_test_guess = 0.5 * (D_RCM_test_min + D_RCM_test_max)
        model.pars.D_RCM_test = D_RCM_test_guess
        model.run(x, t)
        chloride_mean = mh.get_mean(model.C_x_t)

        # compare
        if chloride_mean < cl.mean():
            # narrow the cap
            D_RCM_test_min = max(D_RCM_test_guess, D_RCM_test_min)
        else:
            D_RCM_test_max = min(D_RCM_test_guess, D_RCM_test_max)

        if print_proc:
            print("chloride_mean", chloride_mean)
            print("D_RCM_test", D_RCM_test_guess)
            print("cap", (D_RCM_test_min, D_RCM_test_max))
        count += 1
        if count > max_count:
            print("iteration exceeded max number of iteration: {}".format(count))
            break

    if print_out:
        print("chloride_content:")
        print(
            "model: \nmean:{}\nstd:{}".format(
                mh.get_mean(model.C_x_t), mh.get_std(model.C_x_t)
            )
        )
        print("field: \nmean:{}\nstd:{}".format(cl.mean(), cl.std()))
    return model  # new calibrated obj


def calibrate_chloride_f_group(
    model_raw, t, chloride_content_field, plot=True, print_proc=False
):
    """Use calibrate_chloride_f() to calibrate model to field chloride content at three or more depths, and return the new calibrated model with the averaged D_RCM_0

    Parameters
    ----------
    model_raw : object/instance of Chloride_Model class 
                model object to be calibrated), model_raw.copy() will be used
    chloride_content_field: pandas.DataFrame
                contains field chloride contents at various depths [wt.-%/cement]
    t: int or float
        time [year]

    returns
    -------
    object/instance of Chloride_model class
        a new calibrated model with the averaged calibrated D_RCM_test, and corresponding D_RCM_0 normal distribution array. 
    """
    M_cal_lis = []
    M_cal_new = None
    for i in range(len(chloride_content_field)):
        M_cal = calibrate_chloride_f(
            model_raw,
            chloride_content_field.depth.iloc[i],
            t,
            chloride_content_field.cl.iloc[i],
            print_proc=print_proc,
            print_out=False,
        )
        M_cal_lis.append(M_cal)  # M_cal is a new obj
        print(M_cal.pars.D_RCM_test)

        M_cal_new = model_raw.copy()
        M_cal_new.pars.D_RCM_test = np.mean(
            np.array([M_cal.pars.D_RCM_test for M_cal in M_cal_lis])
        )
        # update the corresponding pars.D_RCM_0 and pars.D_app in the M_cal_new
        M_cal_new.pars.D_app = D_app(t, M_cal_new.pars) 

    if plot:
        Cl_model = [
            mh.get_mean(M_cal_new.run(depth, t))
            for depth in chloride_content_field.depth
        ]
        fig, ax = plt.subplots()
        ax.plot(
            chloride_content_field["depth"],
            chloride_content_field["cl"],
            "--.",
            label="field",
        )
        ax.plot(
            chloride_content_field.depth, Cl_model, "o", alpha=0.5, label="calibrated"
        )
        ax.legend()

    return M_cal_new


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import matplotlib.patches as mpatches

def chloride_year(model, depth, year_lis, plot=True, amplify=80):
    """
    Run the chloride model over a list of time steps and plot results with consistent formatting.
    """
    t_lis = year_lis
    M_cal = model

    M_lis = []
    for t in t_lis:
        M_cal.run(depth, t)
        M_cal.postproc()
        M_lis.append(M_cal.copy())

    if plot:
        fig, [ax1, ax2, ax3] = plt.subplots(
            nrows=3,
            figsize=(8, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [1, 1, 3]},
        )

        # Select representative models for plotting distributions
        indx = np.linspace(0, len(t_lis) - 1, min(6, len(t_lis))).astype(int)[1:]
        M_sel = [M_lis[i] for i in indx]

        # -- Plot P_f --
        ax1.plot([m.t for m in M_lis], [m.pf for m in M_lis], "k--")
        ax1.plot([m.t for m in M_sel], [m.pf for m in M_sel], "k|", markersize=15)
        ax1.set_ylabel(r"$P_f$", fontsize=16)
        ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))

        # -- Plot beta --
        beta_vals = [m.beta_factor for m in M_lis]
        ax2.plot([m.t for m in M_lis], beta_vals, "k--")
        ax2.plot([m.t for m in M_sel], [m.beta_factor for m in M_sel], "k|", markersize=15)
        ax2.set_ylabel(r"$\beta$", fontsize=16)
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))

        # -- Plot distributions --
        ax3.plot(t_lis, [m.pars.C_crit_distrib_param[0] for m in M_lis], "--C0")
        ax3.plot(t_lis, [mh.get_mean(m.C_x_t) for m in M_lis], "--C1")
        for m in M_sel:
            mh.plot_RS(m, ax=ax3, t_offset=m.t, amplify=amplify)

        ax3.set_xlabel("Time [year]", fontsize=16)
        ax3.set_ylabel("Chloride Content [wt-% cement]", fontsize=16)
        ax3.legend(
            handles=[
                mpatches.Patch(color="C0", label="R: critical chloride content", alpha=0.8),
                mpatches.Patch(color="C1", label="S: chloride content at rebar depth", alpha=0.8),
            ],
            loc="upper left",
            fontsize=16
        )

        # Apply consistent tick label size
        for ax in [ax1, ax2, ax3]:
            ax.tick_params(axis="both", which="major", labelsize=16)

        plt.tight_layout()
        plt.show()

    return [m.pf for m in M_lis], [m.beta_factor for m in M_lis]



class ChlorideModel:
    def __init__(self, pars_raw):
        """Initialize the model object.

        Parameters
        ----------
        pars_raw : Param object
            Material and environment parameters.
            A deep copy of `pars_raw` is attached and then updated with derived parameters.
        """
        self.pars = deepcopy(pars_raw)
        self.pars.C_S_dx = C_S_dx(pars_raw)
        self.pars.dx = dx(pars_raw)

    def run(self, x, t):
        """ solve the chloride content at depth x and time t
        Parameters
        ----------
        x : int, float
            Depth x[mm]
        t : float
            Time[year]
        Returns
        -------
        numpy array
            Chloride content at depth x and time t.
        """
        self.C_x_t = chloride_content(x, t, self.pars)
        self.x = x
        self.t = t
        return self.C_x_t

    def postproc(self, plot=False):
        """Postprocess the solved model and attach Pf and beta to the model object.

        Parameters
        ----------
        plot : bool, optional
            If True, plot the R-S curve, by default False.
        """
        sol = mh.pf_RS(
            self.pars.C_crit_distrib_param, self.C_x_t, R_distrib_type="beta", plot=plot
        )
        self.pf = sol[0]
        self.beta_factor = sol[1]
        self.R_distrib = sol[2]
        self.S_kde_fit = sol[3]
        self.S = self.C_x_t

    def calibrate(self, t, chloride_content_field, print_proc=False, plot=True):
        """Return a calibrated model with `calibrate_chloride_f_group()` function.

        Parameters
        ----------
        t : int or float
            Time [year].
        chloride_content_field : pandas DataFrame
            Field chloride contents at various depths [wt.-%/cement].
        print_proc : bool, optional
            If True, print the optimization process, by default False.
        plot : bool, optional
            If True, plot the field vs model comparison, by default True.

        Returns
        -------
        ChlorideModel
            A new calibrated model with the averaged calibrated D_RCM_0.
        """
        model_cal = calibrate_chloride_f_group(
            self, t, chloride_content_field, print_proc=print_proc, plot=plot
        )
        return model_cal

    def copy(self):
        """create a deepcopy of the instance, to preserve the mutable object             
        """
        return deepcopy(self)

    def chloride_with_year(self, depth, year_lis, plot=True, amplify=1):
        """Run the model for a list of time steps.

        Parameters
        ----------
        depth : float
            Depth at which the chloride concrete is calculated, x [mm].
        year_list : list of int or float
            A list of time steps [year].
        plot : bool, optional
            If True, plot the R-S curve, pf, beta with time axis, by default True.
        amplify : int, optional
            A scale parameter adjusting the height of the distribution curve, by default 1.

        Returns
        -------
        numpy array
            Probability of failure (Pf) for each time step.
        numpy array
            Reliability factor (beta) for each time step.
        """
        pf_lis, beta_lis = chloride_year(
            self, depth, year_lis, plot=plot, amplify=amplify
        )
        return np.array(pf_lis), np.array(beta_lis)

