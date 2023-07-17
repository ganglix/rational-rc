"""
**Summary**

2D electrochemical model and its regressed solution \n
icorr = f(moisture, temperature, oxygen availability)

**Field data**

+ Volumetric water content (TDR moisture sensor)

+ corrosion rate (LPR, corrosion sensor) to validate the model

"""


from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import math_helper as mh

def icorr_to_mmpy(icorr):
    """Converts icorr [A/m^2] to corrosion rate [mm/year] using Faraday's laws

    Parameters
    ----------
    icorr : float or numpy array
        Corrosion current density [A/m^2]

    Returns
    -------
    float or numpy array
        Corrosion rate, section loss [mm/year]
    """
    M_Fe = 55.8e-3  # kg/mol
    rho_Fe = 7.874e3  # kg/m^3
    n = 2.0
    F = 96485.33212  # C/mol
    return icorr * M_Fe / (n * F * rho_Fe) * 3600 * 24 * 365 * 1000.0


def mmpy_to_icorr(rate):
    """Converts corrosion rate [mm/year] to icorr [A/m^2] using Faraday's laws

    Parameters
    ----------
    rate : float or numpy array
        Corrosion rate, section loss [mm/year]

    Returns
    -------
    float or numpy array
        Corrosion current density [A/m^2]
    """
    M_Fe = 55.8e-3  # kg/mol
    rho_Fe = 7.874e3  # kg/m^3
    n = 2.0
    F = 96485.33212  # C/mol

    return rate * n * F * rho_Fe / (M_Fe * 3600 * 24 * 365 * 1000.0)


def icorr_base(rho, T, iL, d):  # SI units # regressed model for the ref
    """Calculate averaged corrosion current density over the rebar-concrete interface from resistivity, temperature, limiting current, and cover thickness.

    Parameters
    ----------
    rho : float or numpy array
        Resistivity [ohm.m]
    T : float or numpy array
        Temperature [K]
    iL : float or numpy array 
        Limiting current, oxygen diffusion [A/m^2]
    d : float or numpy array
        Concrete cover depth [m]

    Returns
    -------
    numpy array
        icorr : Corrosion current density, treated as uniform corrosion [A/m^2]

    Notes
    -----
    Reference: Pour-Ghaz, M., Isgor, O. B., & Ghods, P. (2009). The effect of temperature on the corrosion of steel in concrete. Part 1: Simulated polarization resistance tests and model development. Corrosion Science, 51(2), 415–425. https://doi.org/10.1016/j.corsci.2008.10.034
    Parameters from the reference.
    SI units
    """
    # constants
    tau = 1.181102362e-3
    eta = 1.414736274e-5
    c = -0.00121155206
    kappa = 0.0847693074
    lam = 0.130025167
    gamma = 0.800505851
    mu = 1.23199829e-11
    theta = -0.000102886027
    V = 0.475258097
    X = 5.03368481e-7
    nu = 90487
    W = 0.0721605536

    icorr = (
        1
        / (tau * rho ** gamma)
        * (
            eta * T * d ** kappa * iL ** lam
            + mu * T * nu ** (iL ** W)
            + theta * (T * iL) ** V
            + X * rho ** gamma
            + c
        )
    )
    return icorr


def theta2rho_fun(theta_water, a, b):
    """Convert volumetric water content to resistivity using an exponential function.

    Parameters
    ----------
    theta_water : float or numpy array
        Volumetric water content
    a : float
        Regression coefficient
    b : float
        Regression coefficient

    Returns
    -------
    float or numpy array
        Resistivity

    """
    rho = a * theta_water ** b
    return rho


def icorr_f(pars):
    """Calculate the corrosion current density using icorr_base() with volumetric water content.

    Parameters
    ----------
    pars : Param
        An instance of the Param class containing the following attributes:
    pars.theta_water : float
        Volumetric water content
    pars.T : float or numpy array
        Temperature [K]
    pars.iL : float or numpy array 
        Limiting current, oxygen diffusion [A/m^2]
    pars.d : float or numpy array
        Concrete cover depth [m]
    pars.theta2rho_coeff_a : float
        Regression coefficient of theta2rho_fun
    pars.theta2rho_coeff_b : float
        Regression coefficient of theta2rho_fun

    Returns
    -------
    float, numpy array
        icorr : corrosion current density [A/m^2]
    """
    pars.iL = iL_f(pars)
    rho = theta2rho_fun(pars.theta_water, pars.theta2rho_coeff_a, pars.theta2rho_coeff_b)
    icorr = icorr_base(rho, pars.T, pars.iL, pars.d)
    return icorr


def iL_f(pars):
    """calculate O2 limiting current density
    Parameters
    ----------
    pars : instance of Param object
        parameter object that contains the material properties listed in the note.

    Returns
    -------
    float, numpy array
        O2 limiting current density [A/m^2]

    Note
    ----
    intermediate parameters
    
    + z : number of charge, 4 for oxygen
    + delta : thickness of diffusion layer [m]
    + pars.De_O2 : diffusivity [m^2/s]
    + pars.Cs_g : bulk concentration [mol/m^3]
    + pars.epsilon_g : gas phase fraction
    """
    F = 96485.3329  # s*A/mol
    z = 4

    # effective diffusivity averaged for the whole concrete medium
    pars.De_O2 = De_O2_f(pars)
    delta = pars.d
    pars.Cs_g = Cs_g_f()
    pars.epsilon_g = pars.epsilon - pars.theta_water

    # assume quick dissolution between gas and liquid phase
    # liquid phase diffusion is neglected, very slow iL = 0 when epsilon_g = 0

    # pars.epsilon_g * pars.Cs is the concentration of concrete
    iL = z * F * (pars.epsilon_g * pars.De_O2 * pars.Cs_g / delta)
    return iL


def Cs_g_f():
    """Calculate the atmospheric O2 concentration in the gas phase on the boundary [mol/m^3], converted from 20.95% by volume"""
    O2_fraction = 20.95 / 100
    air_molar_vol = 22.4  # [L/mol]
    Cs_g = 1 / air_molar_vol * O2_fraction * 1000  # mol/m^3
    return Cs_g


def De_O2_f(pars):
    """calculate the O2 effective diffusivity of concrete
    Parameters
    ----------
    pars : instance of Param object

    Returns
    -------
    float, numpy array
        O2 effective diffusivity of concrete

    Notes
    -----
    important intermediate Parameters

    + epsilon_p : porosity of hardened cement paste,
    + RH : relative humidity [-%]

    Gas diffusion along the aggregate-paste interface makes up for the lack of diffusion through the aggregate particles themselves.
    Therefore, the value of effective diffusivity is considered herein as a function of the porosity of hardened cement paste. 
    [TODO: add temperature dependence]
    """
    epsilon_p = epsilon_p_f(pars)
    pars.epsilon_p = epsilon_p

    # calculate internal RH with retention curve/ adsoption curve
    waterByMassHCP = theta_water_to_waterByMassHCP(
        pars
    )  # water content g/g hardened cement paste
    pars.waterByMassHCP = waterByMassHCP

    RH = waterByMassHCP_to_RH(pars)
    pars.RH = RH

    # [TODO] D_O2_T0 * np.e**(dU_D/R*(1/T0-1/T)) Pour-Ghaz, M., Burkan Isgor, O., & Ghods, P. (2009). The effect of temperature on the corrosion of steel in concrete. Part 2: Model verification and parametric study. Corrosion Science, 51(2), 426–433. https://doi.org/10.1016/j.corsci.2008.10.036
    De_O2 = 1.92e-6 * epsilon_p ** 1.8 * (1 - RH / 100) ** 2.2  # Papadakis 1991
    return De_O2


def epsilon_p_f(pars):
    """Calculate the porosity of the hardened cement paste from the concrete porosity
    
    Parameters
    ---------
    pars : instance of Param object

    Returns
    -------
    float, numpy array

    Note
    ----
    [TODO: when the concrete porosity is not known, the calculated porosity is time dependent at young age, a function of concrete mix and t]

    """

    if isinstance(int(pars.epsilon), int):  # concrete porosity, epsilon is given
        a_c = pars.a_c  # aggregate cement ratio
        w_c = pars.w_c  # water cement ratio
        rho_c = pars.rho_c  # density of cement
        rho_a = pars.rho_a  # density of aggregate
        rho_w = 1000.0
        epsilon_p = pars.epsilon * (
            1 + (a_c * rho_c / rho_a) / (1 + w_c * rho_c / rho_w)
        )
    elif 1 == 0:
        # use calculation from mix proportioning

        # [TODO: epsilon is time dependent, a function of concrete mix and t]
        epsilon_p = None
    else:
        epsilon_p = None
        print("cement paste porosity, epsilon_p is not configured!")

    return epsilon_p


def calibrate_f(raw_model, field_data):
    """[TODO] A placeholder function for future development. field_data: temperature, theta_water, icorr_list"""
    model = raw_model.copy()
    return model


# RH and water theta is related. Use theoretical model adsorption isotherm or empirical van-Genutchten model
def RH_to_waterByMassHCP(pars):
    """Return water content(g/g hardened cement paste) from RH in pores/environment based on water-cement ratio w_c, cement_type, temperature by using modified BET model

    Note
    ----
    Reference: Xi, Y., Bazant, Z. P., & Jennings, H. M. (1993). Moisture Diffusion in Cementitious Materials Adsorption Isotherms.
    """
    V_m = V_m_f(pars.t, pars.w_c, pars.cement_type)
    pars.V_m = V_m

    C_mean, C = C_f(pars.T)  # mean, distribution sample
    pars.C = C

    k = k_f(C_mean, pars.w_c, pars.t, pars.cement_type)
    pars.k = k

    RH_divided_by_100 = pars.RH / 100

    waterByMassHCP = (
        V_m
        * C
        * k
        * RH_divided_by_100
        / ((1 - k * RH_divided_by_100) * (1 + (C - 1) * k * RH_divided_by_100))
    )

    return waterByMassHCP


def waterByMassHCP_to_RH(pars):
    """Return RH in pores/environment from water content(g/g hardened cement paste) based on water-cement ratio w_c, cement type, temperature,
    a reverse function of RH_to_waterByMassHCP()"""
    V_m = V_m_f(pars.t, pars.w_c, pars.cement_type)
    pars.V_m = V_m

    C_mean, C = C_f(pars.T)  # mean, distribution sample
    pars.C = C

    k = k_f(C_mean, pars.w_c, pars.t, pars.cement_type)
    pars.k = k

    waterByMassHCP = pars.waterByMassHCP

    r1, r2 = mh.f_solve_poly2(
        -(C - 1) * k ** 2, (C - 2 - C * V_m / waterByMassHCP) * k, 1
    )

    if r1.mean() > 0:
        RH_divided_by_100 = r1
    else:
        RH_divided_by_100 = r2

    RH = RH_divided_by_100 * 100
    return RH


def V_m_f(t, w_c, cement_type):
    """Calculate V_m, a BET model parameter.

    Parameters
    ----------
    t : float
        Curing time/concrete age [day]
    w_c : float
        Water-cement ratio
    cement_type : str
        ASTM C150 cement type, see note

    Returns
    -------
    numpy array
        V_m : BET model parameter

    Note
    ----
    ASTM C150 cement type:\n
    Cement Type          Description\n
    Type I            :   Normal\n
    Type II           :   Moderate Sulfate Resistance\n
    Type II (MH)      :   Moderate Heat of Hydration (and Moderate Sulfate Resistance)\n
    Type III          :   High Early Strength\n
    Type IV           :   Low Heat Hydration\n
    Type V            :   High Sulfate Resistance
    """
    if t < 5:
        t = 5

    if w_c < 0.3:
        w_c = 0.3
    if w_c > 0.7:
        w_c = 0.7

    V_ct_cement_type_dict = {
        "Type I": 0.9,
        "Type II": 1,
        "Type III": 0.85,
        "Type IV": 0.6,
    }
    # default value is 0.9, returns default when type is not found
    V_ct = V_ct_cement_type_dict.get(cement_type, 0.9)

    V_m_mean = (0.068 - 0.22 / t) * (0.85 + 0.45 * w_c) * V_ct
    V_m_std = 0.016 * V_m_mean  # COV
    V_m = mh.normal_custom(V_m_mean, V_m_std)
    return V_m


def C_f(T):
    """Return BET model parameter C sampled from a normal distribution.

    Parameters
    ----------
    T : float
        temperature [K]

    Note
    ----
    C varies from 10 to 50. This function is not applicable for elevated temperatures
    """
    C_0 = 855
    C_mean = np.e ** (C_0 / T)
    C_std = C_mean * 0.12  # COV 0.12
    C = mh.normal_custom(C_mean, C_std)
    return C_mean, C


def k_f(C_mean, w_c, t, cement_type):
    """Return BET model parameter k

    Parameters
    ----------
    C_mean : float
        Mean value of BET model parameter C
    w_c : float
        Water-cement ratio
    t : float
        Curing time/concrete age [day]
    cement_type : str
        ASTM C150 cement type

    Returns
    -------
    numpy array
        k : BET model parameter
    """
    if t < 5:
        t = 5

    if w_c < 0.3:
        w_c = 0.3
    if w_c > 0.7:
        w_c = 0.7

    N_ct_cement_type_dict = {
        "Type I": 1.1,
        "Type II": 1,
        "Type III": 1.15,
        "Type IV": 1.5,
    }
    # default value is 1.1, returns default when type is not found
    N_ct = N_ct_cement_type_dict.get(cement_type, 1.1)

    n = (2.5 + 15 / t) * (0.33 + 2.2 * w_c) * N_ct

    k_mean = ((1 - 1 / n) * C_mean - 1) / (C_mean - 1)
    k_std = k_mean * 0.007
    k = mh.normal_custom(k_mean, k_std)
    return k


def waterByMassHCP_to_theta_water(pars):
    """Convert water content from g/g hardened cement paste (HCP) 
    to volumetric in HCP to volumetric in concrete

    Parameters
    ----------
    pars : Param
        An instance of the Param class containing the following attributes:
    pars.waterByMassHCP : float
        Water content by mass in hardened cement paste [g/g]
    pars.rho_c : float
        Density of cement [kg/m^3]
    pars.rho_a : float
        Density of aggregate [kg/m^3]
    pars.a_c : float
        aggregate-cement ratio
    pars.w_c : float
        Water-cement ratio

    Returns
    -------
    float
        theta_water : volumetric water content in concrete
    """
    rho_w = 1000
    waterByMassHCP = pars.waterByMassHCP

    rho_c = pars.rho_c
    rho_a = pars.rho_a
    a_c = pars.a_c
    w_c = pars.w_c
    theta_water_hcp = 1 / (1 + (1 / waterByMassHCP - 1) * rho_w / rho_c)

    theta_water = theta_water_hcp / (
        1 + (a_c * rho_c / rho_a) / (1 + w_c * rho_c / rho_w)
    )
    return theta_water


def theta_water_to_waterByMassHCP(pars):
    """ convert water content from volumetric by concrete to volumetric in HCP to g/g in HCP,
    a reverse function of waterByMassHCP_to_theta_water()
    
    Parameters
    ----------
    pars : Param
        An instance of the Param class containing the following attributes:
    pars.theta_water : float
        volumetric water content in concrete
    pars.rho_c : float
        Density of cement [kg/m^3]
    pars.rho_a : float
        Density of aggregate [kg/m^3]
    pars.a_c : float
        Aggregate-cement ratio
    pars.w_c : float
        Water-cement ratio

    Returns
    -------
    float
        waterByMassHCP : Water content by mass in hardened cement paste [g/g]
    """

    rho_w = 1000
    theta_water = pars.theta_water
    rho_c = pars.rho_c
    rho_a = pars.rho_a
    a_c = pars.a_c
    w_c = pars.w_c

    theta_water_hcp = theta_water * (
        1 + (a_c * rho_c / rho_a) / (1 + w_c * rho_c / rho_w)
    )
    waterByMassHCP = 1 / ((1 / theta_water_hcp - 1) * rho_c / rho_w + 1)
    return waterByMassHCP


class CorrosionModel:
    def __init__(self, pars):
        """Initialize the model with Param object and built-in coefficient"""
        pars.theta2rho_coeff_a = 18.71810174  # [TODO: future updates: uncertainty for a and b]
        pars.theta2rho_coeff_b = -1.37938931
        self.pars = pars
        self.icorr = None
        self.x_loss_rate = None

    def run(self):
        """Solve for icorr and the corresponding section loss rate"""
        self.icorr = icorr_f(self.pars)
        self.x_loss_rate = icorr_to_mmpy(self.icorr)  # [mm/year]

    def calibrate(self, field_data):
        # place holder function for future update#
        # update parameters a and b
        pass

    def copy(self):
        return deepcopy(self)


###### section loss ##########

# output section loss distribution at time t
def x_loss_t_fun(t_end, n_step, x_loss_rate, p_active_t_curve):
    """Return samples of x_loss at a given time t_end. The samples represent the distribution of all possible x_loss with different corrosion history

    Parameters
    ----------
    t_end : float
        Year in which the x_loss is reported
    n_step : int
        Number of time steps
    x_loss_rate : float
        Averaged corrosion rate (x-loss rate)
    p_active_t_curve : tuple
        (t_lis_curve, pf_lis_curve) - Probability curve data

    Returns
    -------
    numpy array
        Section loss at t_end year, a large sample from the distribution
    """

    # probability curve data
    t_lis_curve, pf_lis_curve = p_active_t_curve

    # time step of interest (usually finer step)
    t = np.linspace(0,t_end, n_step)

    # at this_year, (t_end), calculate the accumulated section loss for each time step
    age_lis = t_end - t
    age_lis = age_lis[age_lis>=0]
    x_loss_lis = age_lis * x_loss_rate

    # probability of newly active corrosion onset for each time step
    pf_lis = np.interp(t,t_lis_curve, pf_lis_curve)
    p_corr_onset_lis =  np.diff(pf_lis,prepend=0)

    # sample the accumulated section loss with the the corresponding probability
    from random import choices
    x_loss_at_t = choices(x_loss_lis , p_corr_onset_lis, k = mh.N_SAMPLE)
        
    return np.array(x_loss_at_t)


def x_loss_year(model, year_lis, plot=True, amplify=80):
    """Run x_loss_t_fun() function over time.

    Parameters
    ----------
    model : SectionLossModel object
        An instance of the SectionLossModel class
    year_lis : list
        List of years
    plot : bool, optional
        Flag indicating whether to plot the results, by default True
    amplify : int, optional
        Amplification factor for plotting, by default 80

    Returns
    -------
    list
        List of probabilities of failure (Pf) at each year
    list
        List of reliability factors (beta) at each year
    """
    t_lis = year_lis
    M_cal = model

    M_lis = []
    for t in t_lis:
        M_cal.run(t)
        M_cal.postproc()
        M_lis.append(M_cal.copy())

    if plot:
        fig,[ax1,ax2,ax3] = plt.subplots(nrows = 3, figsize=(8,8),sharex=True,gridspec_kw={'height_ratios': [1,1,3]})
        # plot a few distribution
        indx = np.linspace(0,len(year_lis)-1,min(6,len(year_lis))).astype('int')[1:]
        M_sel = [M_lis[i] for i in indx]

        ax1.plot([this_M.t for this_M in M_lis], [this_M.pf for this_M in M_lis],'k--')
        ax1.plot([this_M.t for this_M in M_sel], [this_M.pf for this_M in M_sel],'k|', markersize=15)
        ax1.set_ylabel('Probability of failure $P_f$')

        ax2.plot([this_M.t for this_M in M_lis], [this_M.beta_factor for this_M in M_lis], 'k--')
        ax2.plot([this_M.t for this_M in M_sel], [this_M.beta_factor for this_M in M_sel], 'k|', markersize=15)
        ax2.set_ylabel(r'Reliability factor $\beta$')

        # plot mean results
        ax3.plot(t_lis, [M.pars.x_loss_limit_mean for M in M_lis], '--C0')
        ax3.plot(t_lis, [mh.get_mean(M.x_loss_t) for M in M_lis], '--C1')
        # plot distribution
        
        for this_M in M_sel:
            mh.plot_RS(this_M, ax=ax3, t_offset=this_M.t, amplify=amplify)

        import matplotlib.patches as mpatches
        R_patch = mpatches.Patch(color='C0', label='R: limit',alpha=0.8)
        S_patch = mpatches.Patch(color='C1', label='S: section loss',alpha=0.8)

        ax3.set_xlabel('Time[year]')
        ax3.set_ylabel('section loss/limit [mm]')
        ax3.legend(handles=[R_patch, S_patch],loc='upper left')

        plt.tight_layout()

    return [this_M.pf for this_M in M_lis], [this_M.beta_factor for this_M in M_lis]


class SectionLossModel:
    def __init__(self, pars):
        self.pars = pars  # pars with user-input

    def run(self, t_end):
        """run model to solve the accumulated section loss at t_end by using x_loss_t_fun()

        Parameters
        ----------
        t_end : int, float
            year
        """
        self.t = t_end
        self.x_loss_t = x_loss_t_fun(t_end, mh.N_SAMPLE, self.pars.x_loss_rate, self.pars.p_active_t_curve)

    def postproc(self, plot=False):
        """calculate the Pf and beta from accumulated section loss and section loss limit

        Parameters
        ----------
        plot : bool, optional
            if true plot the R S curve, by default False
        """
        sol = mh.pf_RS((self.pars.x_loss_limit_mean, self.pars.x_loss_limit_std), self.x_loss_t, plot=plot)
        self.pf = sol[0]
        self.beta_factor = sol[1]
        self.R_distrib = sol[2]
        self.S_kde_fit = sol[3]
        self.S = self.x_loss_t
    
    def copy(self):
        """Return a deep copy of the object
        """
        return deepcopy(self)
    
    def section_loss_with_year(self, year_lis, plot=True, amplify=1):
        """use x_loss_year() to report the accumulated section loss at each time step and
        the corresponding Pf and beta.

        Parameters
        ----------
        year_lis : list
            a list of time step [year]
        plot : bool, optional
            if true, plot the RS, pf, beta with time, by default True
        amplify : int, optional
            scale factor to adjust the height of the distribution curve, by default 1

        Returns
        -------
        tuple
            (pf_list, beta_list)
        """
        pf_lis, beta_lis = x_loss_year(self, year_lis, plot=plot, amplify=amplify)
        return np.array(pf_lis), np.array(beta_lis)
