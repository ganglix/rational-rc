"""
**Summary**

Analytical solution of Fick’s second law under advection zone\n
Modified with material property and exposure environment

+ **Resistance**: 	critical chloride content
+ **Load**: 		chloride content at rebar depth
+ **limit-state**: 	chloride content at rebar depth >= critical chloride content
+ **Field data**: 	chloride content profile

TODO: make t input vectorized 
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.special import erf

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
def Chloride_content(x, t, pars):
    """Chloride_content is the master model function, calculate chloride content at depth x and time t with Fick's 2nd law below the convection zone (x > dx)
    The derived parameters is also calculated within this function.
    
    + Caution: The pars instance is mutable, so a deepcopy of the original instance should be used if the calculation is not intended for "inplace".

    Parameters
    ----------
    x : float, int
        depth at which chloride content C_x_t is reported [mm]
    t : float, int
        time [year]
    pars : instance of param object
        a wrapper of all material and environmental parameters deep-copied from the raw data

    Returns
    -------
    numpy array
        sample of the distribution of the chloride content in concrete at a depth x (surface x=0) at time t [wt-.%/c]
    
    Note
    ----
    intermediate parameters were calculated and attached to pars

    + C_0    : initial chloride content of the concrete [wt-.%/cement]
    + C_S_dx : chloride content at a depth dx and a certain point of time t [wt-.%/cement]
    + dx     : depth of the convection zone (concrete layer, up to which the process of chloride penetration differs from Fick’s 2nd law of diffusion) [mm]
    + D_app  : apparent coefficient of chloride diffusion through concrete [mm^2/year]
    + erf    : imported error function
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
    
    + k_e     : environmental transfer variable [-]
    + D_RCM_0 : chloride migration coefficient [mm^2/year]
    + k_t     : transfer parameter, k_t =1 was set in A_t()[-]
    + A_t     : subfunction considering the 'ageing' [-]
    """
    pars.k_e = k_e(pars)
    pars.D_RCM_0 = D_RCM_0(pars)

    pars.A_t = A_t(t, pars)  # pars.k_t =1 was set in A_t()
    D_app = pars.k_e * pars.D_RCM_0 * pars.k_t * pars.A_t
    return D_app


def k_e(pars):
    """Calculate k_e: environmental transfer variable [-]

    Parameters
    ----------
    pars : instance of param object
        a wrapper of all material and environmental parameters deep-copied from the raw data

        + pars.T_ref  : standard test temperature 293 [K]
        + pars.T_real : temperature of the structural element [K]
        + pars.b_e    : regression variable [K]

    Returns
    -------
    numpy array
        large sample of the distribution of k_e
    """
    pars.T_ref = 293  # K (20°C)
    pars.b_e = b_e()
    k_e = np.e ** (pars.b_e * (1 / pars.T_ref - 1 / pars.T_real))
    return k_e


def b_e():
    """provide the large sample array of b_e : regression variable [K]"""
    b_e = hf.Normal_custom(4800, 700)  # K
    return b_e


def A_t(t, pars):
    """calculate A_t considering the ageing effect

    Parameters
    ----------
    t : int, float
        time [year]
    pars : instance of param object
        a wrapper of all material and environmental parameters deep-copied from the raw data
        
        + pars.concrete_type : string
            Option:

            'Portland cement concrete',

            'Portland fly ash cement concrete',

            'Blast furnace slag cement concrete'

    Returns
    -------
    out : numpy array
        subfunction considering the "ageing"[-]

    Note
    ----
    built-in parameters

        + pars.k_t : transfer parameter, k_t =1 was set for experiment [-]
        + pars.t_0 : reference point of time, 0.0767 [year]
    """
    pars.t_0 = 0.0767  # reference point of time [year]
    # To carry out the quantification of a, the transfer variable k_t was set to k_t = 1:
    pars.k_t = 1
    # a: aging exponent
    a = None
    if pars.concrete_type == "Portland cement concrete":
        # CEM I; 0.40 ≤ w/c ≤ 0.60
        a = hf.Beta_custom(0.3, 0.12, 0.0, 1.0)

    if pars.concrete_type == "Portland fly ash cement concrete":
        # f≥0.20·z;k=0.50; 0.40≤w/c_eqv. ≤0.62
        a = hf.Beta_custom(0.6, 0.15, 0.0, 1.0)

    if pars.concrete_type == "Blast furnace slag cement concrete":
        # CEM III/B; 0.40 ≤ w/c ≤ 0.60
        a = hf.Beta_custom(0.45, 0.20, 0.0, 1.0)

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
                            if true interpolation from existing data table is used
    pars.option.df_D_RCM_0  : pandas.DataFrame
                            experimental data table(cement type, and w/c eqv) for interpolation
    pars.option.cement_type : string
                            select cement type for data interpolation of the df_D_RCM_0,
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
         D_RCM_0_final [mm^2/year]
    """
    if isinstance(pars.D_RCM_test, int) or isinstance(pars.D_RCM_test, float):
        # though test result [m^2/s]
        D_RCM_0_mean = pars.D_RCM_test  # [m^2/s]
        D_RCM_0_std = 0.2 * D_RCM_0_mean
        D_RCM_0_temp = hf.Normal_custom(D_RCM_0_mean, D_RCM_0_std)  # [m^2/s]
    elif pars.option.choose:
        # print 'No test data, interpolate: orientation purpose'
        df = pars.option.df_D_RCM_0
        fit_df = df[pars.option.cement_type].dropna()

        # Curve fit
        x = fit_df.index.astype(float).values
        y = fit_df.values
        # [m^2/s] #interp_extrap_f: defined function
        D_RCM_0_mean = hf.interp_extrap_f(x, y, pars.option.wc_eqv, plot=False) * 1e-12
        D_RCM_0_std = 0.2 * D_RCM_0_mean  # [m^2/s]

        D_RCM_0_temp = hf.Normal_custom(D_RCM_0_mean, D_RCM_0_std)  # [m^2/s]

    else:
        print("D_RCM_0 calculation failed.")
        sys.exit("Error message")

    # unit change [m^2/s] -> [mm^2/year]  final model input
    D_RCM_0_final = 1e6 * 3600 * 24 * 365 * D_RCM_0_temp
    return D_RCM_0_final


# Built-in Data Table for data interpolation

# Data table to interpolate/extrapolate
def load_df_D_RCM():
    """load the data table of the Rapid Chloride Migration(RCM) test
    for D_RCM interpolation.

    Returns
    -------
    pandas.DataFrame
        Data table from experiment
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
    df["CEM_I_42.5_R"] = np.array([np.nan, 8.9, 10.0, 15.8, 17.9, 25.0])
    df["CEM_I_42.5_R+FA"] = np.array([np.nan, 5.6, 6.9, 9.0, 10.9, 14.9])
    df["CEM_I_42.5_R+SF"] = np.array([4.4, 4.8, np.nan, np.nan, 5.3, np.nan])
    df["CEM_III/B_42.5"] = np.array([np.nan, 8.3, 1.9, 2.8, 3.0, 3.4])
    df = df.set_index("wc_eqv")
    return df


def C_eqv_to_C_S_0(C_eqv):
    """ Convert solution chloride content to saturated chloride content in concrete
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
        C_S_0 = hf.interp_extrap_f(x[-5:-1], y[-5:-1], C_eqv, plot=False)
    return C_S_0


# C_S: chloride content at surface = C_S_dx when dx = 0
# C_S_dx: chloride content at subsurface

# Environmental param: Potential chloride impact C_eqv
def C_eqv(pars):
    """Evaluate the Potential chloride impact -> equivalent chloride solution concentration, C_eqv[g/L]
    from the source of 

    1. marine or coastal and/or 
    2. de icing salt 

    It is later used to estimate the boundary condition C_S_dx of contineous exposure or NON-geometry-sensitive intermittent exposure
    
    Parameters
    ----------
    pars : instance of the param object
        a wrapper of all material and environmental parameters deep-copied from the raw data
        See Note for details


    Returns
    -------
    float
          C_eqv, potential chloride impact [g/L]

    Note
    ----
    1. marine or coastal

    + pars.C_0_M : natural chloride content of sea water [g/l]

    2. de-icing salt (hard to quantify)

    + pars.C_0_R : average chloride content of the chloride contaminated water [g/l]
    + pars.n     : average number of salting events per year [-]
    + pars.C_R_i : average amount of chloride spread within one spreading event [g/m2]
    + pars.h_S_i : amount of water from rain and melted snow per spreading period [l/m2]

    C_eqv is used for contineous exposure or NON-geometry-sensitive intermittent exposure.
    For geometry-sensitive condition(road side splash) the tested C_max() should be used.
    """
    C_0_M = pars.C_0_M

    n = pars.n
    C_R_i = pars.C_R_i
    h_S_i = pars.h_S_i

    C_0_R = (n * C_R_i) / h_S_i

    C_eqv = None
    if pars.marine:
        C_eqv = C_0_M + C_0_R
    if not pars.marine:
        C_eqv = C_0_R

    return C_eqv


# exposure condition
def C_S_0(pars):
    """Return (surface) chloride saturation concentration C_S_0 [wt.-%/cement] caused by  C_eqv [g/l]

    Parameters
    ----------
    pars.C_eqv : float
                 calculated with by C_eqv(pars) [g/L]
    pars.C_eqv_to_C_S_0 : global function
        This function is based experiment with the info of

        + binder-specific chloride-adsorption-isotherms
        + the concrete composition(cement/concrete ratio)
        + potential chloride impact C_eqv [g/L]
    
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
    """return the substitute chloride surface concentration, i.e. chloride content just below the advection zone.
    
    Fick's 2nd law applies below the advection zone(depth=dx). No advection effect when dx = 0
    condition considered: continuous/intermittent exposure - 'submerged','leakage', 'spray', 'splash' where C_S_dx = C_S_0.
    The advection depth dx is calculated in the dx() function externally.

    if exposure_condition_geom_sensitive is True: the observed/empirical highest chloride content in concrete C_max is used, C_max is calculated by C_max()

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
            C_S_dx = hf.Normal_custom(C_S_dx_mean, C_S_dx_std, non_negative=True)
        else:
            # intermittent exposure, dx >0, set in dx()
            C_S_dx = pars.C_S_0
    else:
        C_S_dx = None
        logger.warning("C_S_dx calculation failed")
    return C_S_dx


# Convection depth
def dx(pars):
    """return dx : advection depth [mm] dependent on the exposure conditions"""
    condition = pars.exposure_condition
    dx = None
    if condition == "splash":
        # - for splash conditions (splash road environment, splash marine environment)
        dx = hf.Beta_custom(5.6, 8.9, 0.0, 50.0)

    if condition in ["submerged", "leakage", "spray"]:
        # - for submerged marine structures
        # - for leakage due to seawater and constant ground water level
        # - for spray conditions(spray road environment, spray marine environment)
        #   a height of more than 1.50 m above the road (spray zone) no dx develops
        dx = 0.0

    if condition == "other":
        print("to be determined")
        pass
    return dx


#  Chloride surface content CS resp. substitute chloride surface content C_S_dx
def C_max(pars):
    """
    C_max: maximum content of chlorides within the chloride profile [wt.-%/cement]
    calculate from empirical equations or from test data [wt.-%/concrete]

    Parameters
    ----------
    pars.cement_concrete_ratio : float
                      cement/concrete weight ratio, used to convert [wt.-%/concrete] -> [wt.-%/cement]

    pars.C_max_option : string
                        "empirical" - use empirical equation
                        "user_input" - use user input, from test
    pars.x_a : 
        "empirical" option: horizontal distance from the roadside [cm]
    pars.x_h : 
        "empirical" option: height above road surface [cm]

    pars.C_max_user_input : 
        "user_input" option: Experiment-tested maximum chloride content [wt.-%/concrete]

    Returns
    -------
    C_max : float
        maximum content of chlorides within the chloride profile, [wt.-%/cement]

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
    """return the beta distribution parameters for the critical chloride content(total chloride), C_crit [wt.-%/cement]

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
    """calibrate chloride model to field data at one depth at one time.
    Calibrate the chloride model with field chloride test data and return the new calibrated model object/instance
    Optimization method:  Field chloride content at depth x and time t -> find corresponding D_RCM_0(repaid chloride migration diffusivity[m^2/s])

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
    calibrate model to field data at three depths in calibrate_chloride_f_group()
    chloride_content_field[wt.-%/cement] at time t

        + optimizing corresponding D_RCM_0,
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
        chloride_mean = hf.Get_mean(model.C_x_t)
        #         print 'relative tol: ', (D_RCM_test_max - D_RCM_test_min)/ D_RCM_test_guess

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
                hf.Get_mean(model.C_x_t), hf.Get_std(model.C_x_t)
            )
        )
        print("field: \nmean:{}\nstd:{}".format(cl.mean(), cl.std()))
    return model  # new calibrated obj


def calibrate_chloride_f_group(
    model_raw, t, chloride_content_field, plot=True, print_proc=False
):
    """use calibrate_chloride_f() to calibrate model to field chloride content at three or more depths, and return the new calibrated model with the averaged D_RCM_0

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
        a new calibrated model with the averaged calibrated D_RCM_0
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

    if plot:
        Cl_model = [
            hf.Get_mean(M_cal_new.run(depth, t))
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


def chloride_year(model, depth, year_lis, plot=True, amplify=80):
    """run model over a list of time steps"""
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
        # plot a few distribution
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
        ax3.plot(t_lis, [M.pars.C_crit_distrib_param[0] for M in M_lis], "--C0")
        ax3.plot(t_lis, [hf.Get_mean(M.C_x_t) for M in M_lis], "--C1")
        # plot distribution
        for this_M in M_sel:
            hf.RS_plot(this_M, ax=ax3, t_offset=this_M.t, amplify=amplify)

        import matplotlib.patches as mpatches

        R_patch = mpatches.Patch(
            color="C0", label="R: critical chloride content", alpha=0.8
        )
        S_patch = mpatches.Patch(
            color="C1", label="S: chloride content at rebar depth", alpha=0.8
        )

        ax3.set_xlabel("Time[year]")
        ax3.set_ylabel("Chloride content[wt-% cement]")
        ax3.legend(handles=[R_patch, S_patch], loc="upper left")

        plt.tight_layout()
    return [this_M.pf for this_M in M_lis], [this_M.beta_factor for this_M in M_lis]


class Chloride_Model:
    def __init__(self, pars_raw):
        """initialize the model object

        Parameters
        ----------
        pars_raw : Param object
            material and environment parameters
            attached a deepcopy of pars_raw with user-input, then update the copy with derived parameters
        """
        #
        self.pars = deepcopy(pars_raw)
        self.pars.C_S_dx = C_S_dx(pars_raw)
        self.pars.dx = dx(pars_raw)

    def run(self, x, t):
        """ solve the chloride content at depth x and time t
        Parameters
        ----------
        x : int, float
            depth x[mm]
        t : float
            time[year]
            
        """
        self.C_x_t = Chloride_content(x, t, self.pars)
        self.x = x
        self.t = t
        return self.C_x_t

    def postproc(self, plot=False):
        """postprocess the solved model and attach the Pf and beta to the model object

        Parameters
        ----------
        plot : bool, optional
            if true, plot the R S curve, by default False
        """
        sol = hf.Pf_RS(
            self.pars.C_crit_distrib_param, self.C_x_t, R_distrib_type="beta", plot=plot
        )
        self.pf = sol[0]
        self.beta_factor = sol[1]
        self.R_distrib = sol[2]
        self.S_kde_fit = sol[3]
        self.S = self.C_x_t

    def calibrate(self, t, chloride_content_field, print_proc=False, plot=True):
        """return a calibrated model with calibrate_chloride_f_group() function

        Parameters
        ----------
        t : int, float
            time [year]
        chloride_content_field : pandas.DataFrame
            contains field chloride contents at various depths [wt.-%/cement]
        print_proc : bool, optional
            if true, print the optimization process, by default False
        plot : bool, optional
            if true, plot the field vs model comparison, by default True

        Returns
        -------
        instance of Chloride_Model object
            a new calibrated model with the averaged calibrated D_RCM_0
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
        """chloride_with_year runs the model for a list of time steps

        Parameters
        ----------
        depth : float
            depth at which the chloride concrete is calculated, x[mm]
        year_lis : list
            a list of time steps [year]
        plot : bool, optional
            if true, plot the R S curve, pf, beta with time axis, by default True
        amplify : int, optional
            a scale parameter adjusting the hight of the distribution curve, by default 80

        Returns
        -------
        tuple
            (pf list, beta list)
        """
        pf_lis, beta_lis = chloride_year(
            self, depth, year_lis, plot=plot, amplify=amplify
        )
        return np.array(pf_lis), np.array(beta_lis)

