import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import logging

# Declare first, as it provides global default value for helper functions
N_SAMPLE = int(1e5)


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


# master helper functions

# Helper function
def dropna(x):
    return x[~np.isnan(x)]


def Get_mean(x):
    """get mean ignoring nans"""
    x = x[~np.isnan(x)]  # remove nans
    return x.mean()


def Get_std(x):
    """get standard deviation ignoring nans"""
    x = x[~np.isnan(x)]  # remove nans
    return x.std()


def Hist_custom(S):
    """plot histogram with N_SAMPLE//100 bins ignoring nans"""
    S_dropna = S[~np.isnan(S)]
    fig, ax = plt.subplots()
    ax.hist(
        S_dropna, bins=min(N_SAMPLE // 100, 100), density=True, alpha=0.5, color="C0"
    )


# Sampler updated
def Normal_custom(m, s, n_sample=N_SAMPLE, non_negative=False, plot=False):
    """ Sampling from a normal distribution

    Parameters
    ----------
    m : int or float
        mean
    s : int or float
        standard deviation
    n_sample : int
        sample number, default is a Global var N_SAMPLE
    non_negative: bool
        if true, return truncated distribution with no negatives, default is False
    plot : bool
        default is False

    Returns
    -------
    out : numpy array
    """
    x = np.random.normal(loc=m, scale=s, size=n_sample)
    if non_negative:
        x = stats.truncnorm.rvs(
            (0 - m) / s, (np.inf - m) / s, loc=m, scale=s, size=n_sample
        )
    if plot:
        fig, ax = plt.subplots()
        ax.hist(x)
        plt.show()
    return x


def Beta_custom(m, s, a, b, n_sample=N_SAMPLE, plot=False):
    """ draw samples from a general beta distribution described by mean, std and lower and upper   bounds
    X~General Beta(a,b, loc = c, scale = d)
    Z~std Beta(alpha, beta)

    X = c + d*Z
    E(X) = c + d * E(Z)
    var(X) = d^2 * var(Z)

    Parameters
    ----------
    m : mean
    s : standard deviation
    a : lower bound, not shape param a(alpha)
    b : upper bound, not shape param b(beta)
    n_sample: int
        sample number
    plot : bool
        default is False

    Returns
    -------
    out : numpy array
    """
    # location:c and scale:d for General Beta (standard Beta range [0,1])
    c = a
    d = b - a

    # mean and variance for
    mu = (m - c) / d
    var = s ** 2 / d ** 2

    # shape params for Z~standard beta
    alpha = ((1 - mu) / var - 1 / mu) * mu ** 2
    beta = alpha * (1 / mu - 1)
    z = np.random.beta(alpha, beta, size=n_sample)

    # transfer back to General Beta
    x = c + d * z

    if plot:
        fig, ax = plt.subplots()
        ax.hist(x)
        print(x.mean(), x.std())
        plt.show()
    return x


def interp_extrap_f(x, y, x_find, plot=False):
    """interpolate or extrapolate value from an array with fitted2-deg or 3-deg polynomial

    Parameters
    ----------
    x : array-like
        varible
    y : array-like
        function value
    x_find : int or float or array-like
        look-up x
    plot : bool
        plot curve fit and data points, default if false

    Returns
    -------
    int or float or array-like
        inter/extrapolated value(s), raise warning when extrapolation is used
    """

    def func2(x, a, b, c):
        # 2-order polynomial
        return a * (x ** 2) + b * (x ** 1) + c

    def func3(x, a, b, c, d):
        # 3-order polynomial
        return a * (x ** 2) + b * (x ** 2) + c * x + d

    if (x_find < x.min()).any() or (x_find > x.max()).any():
        logger.warning("Warning: extrapolation used")

    from scipy.optimize import curve_fit

    # Initial parameter guess, just to kick off the optimization
    if len(y) > 3:
        logger.debug("use func3: 3-deg polynomial")
        guess = (0.5, 0.5, 0.5, 0.5)
        popt, _ = curve_fit(func3, x, y, p0=guess)
        y_find = func3(x_find, *popt)
        if plot:
            fig, ax = plt.subplots()
            ax.plot(x, y, ".", label="table")
            _plot_data = np.linspace(x.min(), x.max(), 100)
            ax.plot(_plot_data, func3(_plot_data, *popt), "--")
            ax.plot(
                x_find, y_find, "x", color="r", markersize=8, label="interp/extrap data"
            )
            ax.legend()
            plt.show()

    elif len(y) <= 3:
        logger.debug("use func2: 2-deg polynomial")
        guess = (0.5, 0.5, 0.5)
        popt, _ = curve_fit(func2, x, y, p0=guess)
        y_find = func2(x_find, *popt)
        if plot:
            fig, ax = plt.subplots()
            ax.plot(x, y, ".", label="table")
            _plot_data = np.linspace(x.min(), x.max(), 100)
            ax.plot(_plot_data, func2(_plot_data, *popt), "--")
            ax.plot(
                x_find, y_find, "x", color="r", markersize=8, label="interp/extrap data"
            )
            ax.legend()
            plt.show()
    else:
        y_find = None
    return y_find


def find_similar_group(item_list, similar_group_size=2):
    """find similar sublist of similar_group_size from a item_list"""
    from itertools import combinations

    combos = np.array(list(combinations(item_list, similar_group_size)))
    ind_min = combos.std(axis=1).argmin()
    similar_group = combos[ind_min].tolist()
    return similar_group


def sample_integral(Y, x):
    """integrate Y over x, where every Y point is a bunch of distribution samples,

    Parameters
    ----------
    Y : numpy array
        2D
        column: y data points
        row: samples for y point
    x : numpy array

    Returns
    -------
    int_y_x : numpy array
            integral of y over x for all sampled data
    """
    from scipy.integrate import simps

    n, _ = Y.shape
    if n != len(x):
        raise Exception("Y does not have the same number of data points as x")
    int_y_x = simps(Y, x, axis=0)
    return int_y_x


def f_solve_poly2(a, b, c):
    r1 = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
    r2 = (-b - (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
    return r1, r2


# helper function
def Fit_distrib(s, fit_type="kernel", plot=False, xlabel="", title="", axn=None):
    """fit data to a probability distribution function(parametric or numerical)
    and return a continuous random variable or a random variable represented by Gaussian kernels
    parametric : normal
    numerical : Gaussian kernels

    Parameters
    ----------
    s : array-like
        sample data
    fit_type : string
        fit type keywords, 'kernel', 'normal'
    plot : bool
        create a plot with histogram and fitted pdf curve

    Returns
    -------
    out : continuous random variable : stats.norm(loc = mu, scale = sigma)
              when parametric normal is used
          Gaussian kernel random variable : (stats.gaussian_kde)
              when kernel is used
    """
    mu = None
    sigma = None
    kde = None
    if fit_type == "normal":
        # parametric, fit normal distribution
        logger.debug("parametric, fit normal distribution")

        # Fit a curve to the variates  mu is loc sigma is scale
        mu, sigma = stats.norm.fit(s, floc=s.mean())

    elif fit_type == "kernel":
        # non-parametric, this creates the kernel, given an array it will estimate the probability over that values
        logger.debug("non-parametric kernel fit")
        s_dropna = s[~np.isnan(s)]  # remove nans
        # bandwidth selection:  gaussian_kde uses a rule of thumb, the default is Scott’s Rule.
        kde = stats.gaussian_kde(s_dropna)
    else:
        raise Exception("fit_type is not set correctly")

    if plot:
        if axn is None:
            axn = plt.gca()
        n = min(len(s) // 100, 100)  # bin size
        dist_space = np.linspace(min(s), max(s), 100)
        axn.hist(s, bins=n, density=True)

        # plot pdf
        if fit_type == "normal":
            # probability distribution
            pdf = stats.norm.pdf(dist_space, mu, sigma)
            axn.plot(dist_space, pdf, label="normal")

        if fit_type == "kernel":
            pdf_kde = kde(dist_space)
            axn.plot(dist_space, pdf_kde, label="kernel")

        axn.set_xlabel(xlabel)
        axn.set_ylabel("distribution density")
        axn.legend(loc="upper right")
        axn.set_title(title)
    # return
    if fit_type == "normal":
        return stats.norm(loc=mu, scale=sigma)
    if fit_type == "kernel":
        return kde


def Pf_RS(R_info, S, R_distrib_type="normal", plot=False):  # updated!
    """Calculate the probability of failure  Pf = P(R-S<0), given the R(resistance) and S(load)
       with three three methods and use method 3) if checked OK with the other two
           1) crude monte carlo
           2) numerical integral of g kernel fit
           3) R S integral: ('$\int\limits_{-\infty}^{\infty} F_R(x)f_S(x)dx$')
       reliability index(beta factor) is calculated with simple 1st order g.mean()/g.std()

    Parameters
    ----------
    R_info : tuple or numpy array
             distribution of Resistance, e.g. cover thickness, critical chloride content, tensile strength
             can be array or distribution parameters
             R_distrib_type='normal' -> tuple(m,s) for normal m: mean s: standard deviation
             R_distrib_type='normal' -> tuple(m,s,a,b) for (General) beta distribution
                             m: mean, s: standard deviation a,b : lower, upper bound
             R_distrib_type='normal' -> array: for not-determined distribution, will be treated numerically(R S integral is not applied )

    S : numpy array
        distribution of load, e.g. carbonation depth, chloride content, tensile stress
        the distribution type is calculated S is usually not determined, can vary a lot in different cases, therefore fitted with kernel

    R_distrib_type : string
        'normal', 'beta', 'array'

    Returns
    -------
    out = tuple
        (probability of failure, reliability index)

    Notes
    -----
    For R as arrays R S integral is not applied
    R S integration method: $P_f = P(R-S<=0)=\int\limits_{-\infty}^{\infty}f_S(y) \int\limits_{-\infty}^{y}f_R(x)dxdy$
    the dual numerical integration seems too computationally expensive, so consider fit R to analytical distribution in the future versions

    """
    from scipy import integrate

    R, pf_RS = (None, None)

    S_kde_fit = Fit_distrib(S, fit_type="kernel")
    S_dropna = S[~np.isnan(S)]

    if R_distrib_type == "normal":
        # R = (mu, std)
        (m, s) = R_info
        R_distrib = stats.norm(m, s)
        R = R_distrib.rvs(size=N_SAMPLE)

        # Calculate probablility of failure
        #     $P_f = P(R-S<=0)=\int\limits_{-\infty}^{\infty} F_R(x)f_S(x)dx$
        pf_RS = integrate.quad(
            lambda x: R_distrib.cdf(x) * S_kde_fit(x)[0], 0, S_dropna.max()
        )[0]

    elif R_distrib_type == "beta":
        # R = (m, s, a, b) a, b are lower and upper bound
        (m, s, a, b) = R_info

        # location:c and scale:d for General Beta (standard Beta range [0,1])
        # calculate loc and scale
        c = a
        d = b - a

        # mean and variance for
        mu = (m - c) / d
        var = s ** 2 / d ** 2

        # shape params for Z~standard beta
        alpha = ((1 - mu) / var - 1 / mu) * mu ** 2
        beta = alpha * (1 / mu - 1)

        R_distrib = stats.beta(alpha, beta, c, d)
        R = R_distrib.rvs(size=N_SAMPLE)

        # Calculate probability of failure
        #     $P_f = P(R-S<=0)=\int\limits_{-\infty}^{\infty} F_R(x)f_S(x)dx$
        pf_RS = integrate.quad(
            lambda x: R_distrib.cdf(x) * S_kde_fit(x)[0], 0, S_dropna.max()
        )[0]

    elif R_distrib_type == "array":
        # dual numerical integration is too expensive, consider fit R to analytical distribution!!!!!!!!!!!!!!!!!!!!!!!!!!
        # plot condition to be fixed !!!!!!!!!!!!!!!!!!!!!!

        #         # use R array
        #         R_kde_fit = Fit_distrib(R, fit_type='kernel')
        #         R_dropna = R[~np.isnan(R)]
        #         # $P_f = P(R-S<=0)=\int\limits_{-\infty}^{\infty}f_S(y) \int\limits_{-\infty}^{y}f_R(x)dxdy$

        #         def R_cdf_S_pdf(x, R_kde_fit, S_kde_fit):
        #             R_cdf = integrate.quad(lambda z: R_kde_fit(z)[0],0,x)[0] # kde_fit returns ([array needed]). therefore use lamda z kde(z)[0]
        #             S_pdf = S_kde_fit(x)[0]
        #             return R_cdf*S_pdf

        #         pf_RS = integrate.quad(R_cdf_S_pdf,0,S_dropna.max(), args=(R_kde_fit, S_kde_fit))[0]
        R_distrib = None
    else:
        R_distrib = None
        pass

    # compare with
    # numerical g
    g = R - S
    g = g[~np.isnan(g)]
    # numerical kernel fit
    g_kde_fit = Fit_distrib(g, fit_type="kernel", plot=False)
    pf_kde = integrate.quad(g_kde_fit, g.min(), 0)[0]
    pf_sample = len(g[g <= 0]) / len(g)
    beta_factor = g.mean() / g.std()  # first order

    # check for tiny tail
    if pf_sample < 1e-10:
        print("warning: very small Pf ")
        logger.warning("warning: very small Pf ")

    # check if pf_RS is the pf (should be)
    best_2_of_3 = find_similar_group([pf_sample, pf_kde, pf_RS], similar_group_size=2)
    if pf_RS not in best_2_of_3:
        logger.warning("warning: pf_RS is not used, double check")
        logger.warning(
            "Pf(g = R-S < 0) from various methods\n    sample count: {}\n    g integral: {}\n    R S integral: {}\n    beta_factor: {}".format(
                pf_sample, pf_kde, pf_RS, beta_factor
            )
        )

    logger.info(
        "Pf(g = R-S < 0) from various methods\n    sample count: {}\n    g integral: {}\n    R S integral: {}\n    beta_factor: {}".format(
            pf_sample, pf_kde, pf_RS, beta_factor
        )
    )

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
            bins=min(N_SAMPLE // 100, 100),
            density=True,
            alpha=0.5,
            color="C0",
            label="R",
        )

        # S
        S_plot = np.linspace(S_dropna.min(), S_dropna.max(), 100)
        ax1.plot(S_plot, S_kde_fit(S_plot), color="C1", alpha=1)
        ax1.hist(
            S_dropna,
            bins=min(N_SAMPLE // 100, 100),
            density=True,
            alpha=0.5,
            color="C1",
            label="S",
        )

        ax1.set_title(
            "S: mean = {:.1f} stdev = {:.1f}".format(S_dropna.mean(), S_dropna.std())
        )
        ax1.legend()
        plt.tight_layout()

        # plot g
        g_plot = np.linspace(g.min(), g.max(), 100)
        ax2.plot(g_plot, g_kde_fit(g_plot), color="C2", alpha=1)

        ax2.hist(
            g,
            density=True,
            bins=min(N_SAMPLE // 100, 100),
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
            s=r"${\mu}_g$",
            xy=(0, g.mean()),
            xytext=(g.mean(), g_kde_fit(0)[0]),
            va="center",
        )
        ax2.legend()
        ax2.set_title("Limit-state P(g<0)={}".format(pf_RS))
        plt.show()

    return pf_RS, beta_factor, R_distrib, S_kde_fit


def RS_plot(model, ax=None, t_offset=0, amplify=1):  # updated!
    """plot R S distribution vertically at a time to an axis

    Parameters
    ----------
    model.R_distrib : scipy.stats._continuous_distns, normal or beta
                      calculated in Pf_RS() through model.postproc()
    model.S_kde_fit : stats.gaussian_kde
                      calculated in Pf_RS() through model.postproc()
                      distribution of load, e.g. carbonation depth, chlride content, tensile     stress. The distrubtion type is calculated S is usually not determined, can vary a lot in different cases, therefore fitted with kernel

    model.S : numpy array
              load, e.g. carbonation depth, chloride content, tensile stress
    ax : axis
    t_offset : time offset to move the plot along the t-axis. default is zero
    amplify : scale the height of the pdf plot
    """

    R_distrib = model.R_distrib
    S_kde_fit = model.S_kde_fit
    S = model.S

    S_dropna = S[~np.isnan(S)]
    # Plot R S
    R = R_distrib.rvs(size=N_SAMPLE)

    if ax is None:
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
    # S  avoid ploting large S with very small probability
    S_plot = np.linspace(S_dropna.min(), min(5 * S_dropna.mean(), S_dropna.max()), 100)
    ax.plot(S_kde_fit(S_plot) * amplify + t_offset, S_plot, color="C1", alpha=1)
    ax.fill_betweenx(
        S_plot,
        t_offset,
        S_kde_fit(S_plot) * amplify + t_offset,
        color="C1",
        alpha=0.5,
        label="S",
    )


# helper function
def find_mean(val, s, confidence_one_tailed=0.95):
    """return the mean value of a unknown normal distribution
    based on the given value at a known one-tailed confidence level(default 95%)

    Parameters
    ----------
    val : float
         cut-off value
    s : standard deviation
    confidence_one_tailed : confidence level

    Returns
    -------
    mean : mean value of the unknown distribution
    """

    def func(m, s, val, confidence_one_tailed):
        """object function to be solved"""
        norm = stats.norm(m, s)
        cutoff = norm.cdf(val)
        return cutoff - (1 - confidence_one_tailed)

    from scipy.optimize import fsolve

    mean = fsolve(func, x0=val, args=(s, val, confidence_one_tailed))[
        0
    ]  # use val as initial guess
    return mean
