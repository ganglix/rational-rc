"""
**Summary**

A statistical model is used to predict the probability of failure for the membrane. 

+ **Resistance**: membrane service life

+ **Load**: age 

+ **limit-state**: age >= service life. 

"""

import matplotlib.pyplot as plt
from copy import deepcopy
from scipy import stats
import numpy as np
import helper_func as hf

# special functions for this module


def Pf_RS_special(R_info, S, R_distrib_type="normal", plot=False):
    """special case of helper_fuc.Pf_RS, here the "load" S is a number and it calculates the probability of failure  Pf = P(R-S<0), given the R(resistance) and S(load)
    with three three methods and use method 3 if it is checked "OK" with the other two

    1. crude monte carlo  
    2. numerical integral of g kernel fit
    3. R S integral: $F_R(S)$, reliability index(beta factor) is calculated with simple 1st order g.mean()/g.std()
    
    Parameters
    ----------
    R_info : tuple
        distribution of Resistance, for this specicial case, the membrane service life.
        R_distrib_type='normal' -> tuple(m,s) for normal m: mean s: standard deviation
        other distribution form will be ignored.

    S : numpy array
        distribution of load, for this special case, the membrane age.

    R_distrib_type : str, optional
        by default 'normal'

    plot : bool, optional
        plot distribution, by default False

    Returns
    -------
    tuple
        (probability of failure, beta factor)

    Note
    ----
    R_info only supports two-parameter normal distribution.
    """
    from scipy import integrate

    if isinstance(int(S), int):
        if R_distrib_type == "normal":
            # R = (mu, std)
            (m, s) = R_info
            R_distrib = stats.norm(m, s)
            R = R_distrib.rvs(size=hf.N_SAMPLE)

            # Calculate probablility of failure
            pf_RS = R_distrib.cdf(S)
        else:
            R = None
            pf_RS = None
            R_distrib = None
            print("R is not configured because R_distrib is not normal")
    else:
        R = None
        S = None
        pf_RS = None
        R_distrib = None
        print("S is not configured")

    # compare with
    # numerical g
    g = R - S
    g = g[~np.isnan(g)]
    # numerical kernel fit
    g_kde_fit = hf.Fit_distrib(g, fit_type="kernel", plot=False)

    pf_kde = integrate.quad(g_kde_fit, g.min(), 0)[0]
    pf_sample = len(g[g < 0]) / len(g)
    beta_factor = g.mean() / g.std()  # first order

    if plot:
        print("Pf(g = R-S < 0) from various methods")
        print("    sample count: {}".format(pf_sample))
        print("    g integral: {}".format(pf_kde))
        print("    R S integral: {}".format(pf_RS))
        # printmd('$\int\limits_{-\infty}^{\infty} F_R(x)f_S(x)dx$')
        print("    beta_factor: {}".format(beta_factor))

        # Plot R S
        fig, [ax1, ax2] = plt.subplots(ncols=2, figsize=(10, 3))
        # R
        R_plot = np.linspace(R.min(), R.max(), 100)
        ax1.plot(R_plot, R_distrib.pdf(R_plot), color="C0")
        ax1.hist(
            R,
            bins=min(hf.N_SAMPLE // 100, 100),
            density=True,
            alpha=0.5,
            color="C0",
            label="R",
        )

        # S updated
        ax1.vlines(
            x=S, ymin=0, ymax=R_distrib.pdf(R_info[0]), color="C1", alpha=1, label="S"
        )  # updated

        ax1.set_title("S: {:.1f}".format(S))
        ax1.legend()
        plt.tight_layout()

        # plot g
        g_plot = np.linspace(g.min(), g.max(), 100)
        ax2.plot(g_plot, g_kde_fit(g_plot), color="C2", alpha=1)

        ax2.hist(
            g,
            density=True,
            bins=min(hf.N_SAMPLE // 100, 100),
            color="C2",
            alpha=0.5,
            label="g=R-S",
        )
        ax2.vlines(x=0, ymin=0, ymax=g_kde_fit(0)[0], linestyles="--", alpha=0.5)
        ax2.vlines(
            x=g.mean(), ymin=0, ymax=g_kde_fit(g.mean())[0], linestyles="--", alpha=0.5
        )
        #         ax.annotate(s='', xy=(0,g_kde_fit(0)[0]), xytext=(g.mean(),g_kde_fit(0)[0]),
        #                     arrowprops={'arrowstyle': '<->'},va='center')
        ax2.annotate(
            s=r"$\{mu}_g$",
            xy=(0, g.mean()),
            xytext=(g.mean(), g_kde_fit(0)[0]),
            va="center",
        )
        ax2.legend()
        ax2.set_title("Limit-state P(g<0)={}".format(pf_RS))
        plt.show()

    return pf_RS, beta_factor, R_distrib


def RS_plot_special(model, ax=None, t_offset=0, amplify=1):  # updated!
    """plot R S distribution vertically at a time to an axis special case: S is a number.

    Parameters
    ----------
    model : model object instance
        
        + model.R_distrib : scipy.stats._continuous_distns, normal or beta [calculated in Pf_RS() through model.postproc()]
        + model.S : single number for this special case

    ax : axes
        subplot axis

    t_offset : int, float
        time offset to move the plot along the t-axis. default is zero
    
    amplify : int
        scale the height of the pdf plot
    """

    R_distrib = model.R_distrib
    S = model.S

    S_dropna = S[~np.isnan(S)]
    # Plot R S
    R = R_distrib.rvs(size=hf.N_SAMPLE)

    if ax == None:
        ax = plt.gca()
    # R
    R_plot = np.linspace(R.min(), R.max(), 100)
    ax.plot(R_distrib.pdf(R_plot) * amplify + t_offset, R_plot, color="C0")
    ax.fill_betweenx(
        R_plot,
        t_offset,
        R_distrib.pdf(R_plot) * amplify + t_offset,
        color="C0",
        alpha=0.5,
        label="R",
    )
    # S
    S_plot = np.linspace(S_dropna.min(), S_dropna.max(), 100)
    ax.hlines(
        y=S,
        xmin=0 * amplify + t_offset,
        xmax=R_distrib.pdf(R.mean()) * amplify + t_offset,
        color="C1",
        alpha=1,
        label="S",
    )


# model function
def membrane_age(t):
    """return the membrane age as the "resistance"

    Parameters
    ----------
    t : int, float
        membrane age

    Returns
    -------
    int, float
        membrane age
    
    Note
    ----
        This function is a placeholder for more complex age input
    """
    return t


def membrane_life(pars):
    """calculate the mean value of the service life from the manufacture's service life label(eg. 30 years with 95% confidence)
    with the given standard deviation

    Parameters
    ----------
    pars : parameter object instance
        raw parameters
        pars.life_product_label_life
        pars.life_confidence
        pars.life_std

    Returns
    -------
    float
        service life mean value
    """
    life_mean = hf.find_mean(
        # find_mean is a helper function in helper_func.py
        val=pars.life_product_label_life,
        s=pars.life_std,
        confidence_one_tailed=pars.life_confidence,
    )
    return life_mean


# calibrate Resistance to match the probability
def calibrate_f(
    model_raw, t, membrane_failure_ratio_field, tol=1e-6, max_count=100, print_out=True
):
    """calibrate membrane model to field condition by finding the corresponding membrane service life std that matches the failure ratio in the field

    Parameters
    ----------
    model_raw : model instance
        model to be calibrated
    t : int, float
        membrane age when membrane failure rate is surveyed [year]
    membrane_failure_ratio_field : float
        failure rate e.g. 0.1 for 10%
    tol : float, optional
        optimization tolerance, by default 1e-6
    max_count : int, optional
        optimization max iteration number, by default 100
    print_out : bool, optional
        if True print out model vs field compare, by default True

    Returns
    -------
    membrane model object instance
        calibrated model
    """
    
    model = model_raw.copy()
    std_min = 0.0
    std_max = 100.0  # year, unrealistic large safe ceiling
    # optimization
    count = 0
    while std_max - std_min > tol:
        # update guess
        std_guess = 0.5 * (std_min + std_max)
        model.pars.life_std = std_guess
        model.run(t)
        model.postproc()

        # compare
        if model.pf < membrane_failure_ratio_field:
            # narrow the cap
            std_min = max(std_guess, std_min)
        else:
            std_max = min(std_guess, std_max)

        # print([std_min, std_max])
        count += 1
        if count > max_count:
            break

    if print_out:
        print("probability of failure:")
        print("model: {}".format(model.pf))
        print("field: {}".format(membrane_failure_ratio_field))
    return model


def membrane_failure_year(model, year_lis, plot=True, amplify=30):
    """membrane_failure_year: run model over a list of time steps

    Parameters
    ----------
    model : class instance
        Membrane_model class instance
    year_lis : list, array-like
        a list of time steps
    plot : bool, optional
        if True, plot the Pf, beta, R S distribution, by default True
    amplify : int, optional
        the arbitrary comparable size of the distribution curve, by default 80

    Returns
    -------
    tuple
        (pf list , beta list)
    """
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
        ax3.plot(t_lis, [M.pars.life_mean for M in M_lis], "--C0")
        ax3.plot(t_lis, [M.age for M in M_lis], "--C1")
        # plot distribution
        for this_M in M_sel:
            RS_plot_special(this_M, ax=ax3, t_offset=this_M.t, amplify=amplify)

        import matplotlib.patches as mpatches

        R_patch = mpatches.Patch(color="C0", label="R: membrane life", alpha=0.8)
        S_patch = mpatches.Patch(color="C1", label="S: age", alpha=0.8)

        ax3.set_xlabel("Time[year]")
        ax3.set_ylabel("age/membrane life [year]")
        ax3.legend(handles=[R_patch, S_patch], loc="upper left")
        plt.tight_layout()

    return [this_M.pf for this_M in M_lis], [this_M.beta_factor for this_M in M_lis]


class Membrane_Model:
    def __init__(self, pars):
        """initialize the object with raw parameter object (pars) and mean membrane life"""
        self.pars = pars
        self.pars.life_mean = membrane_life(self.pars)

    def run(self, t):
        """attach the resistance: membrane age"""
        self.t = t
        self.age = membrane_age(t)

    def postproc(self, plot=False):
        """solve pf, beta, attach R distribution with plot option"""
        sol = Pf_RS_special(
            (self.pars.life_mean, self.pars.life_std),
            self.age,
            R_distrib_type="normal",
            plot=plot,
        )
        self.pf = sol[0]
        self.beta_factor = sol[1]
        self.R_distrib = sol[2]
        self.S = self.age

    def membrane_failure_with_year(self, year_lis, plot=True, amplify=80):
        """solve pf, beta at a list of time steps with plot option"""
        pf_lis, beta_lis = membrane_failure_year(
            self, year_lis, plot=plot, amplify=amplify
        )
        return np.array(pf_lis), np.array(beta_lis)

    def copy(self):
        """create a deepcopy"""
        return deepcopy(self)

    def calibrate(self, membrane_age_field, membrane_failure_ratio_field):
        """calibrate membrane model to field condition

        Parameters
        ----------
        membrane_age_field : float, int
            membrane age when membrane failure rate is surveyed
        membrane_failure_ratio_field : float
            failure rate e.g. 0.1 for 10%

        Returns
        -------
        membrane model object instance
            calibrated model
        """
        M_cal = calibrate_f(self,membrane_age_field, membrane_failure_ratio_field)
        return M_cal