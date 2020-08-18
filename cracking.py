"""
# Cracking V0.2.3
## Input
+ raw
section loss

## Output
internal stress strain through the concrete cover and the location of the crack tip
critical section loss to cause crack on the cover surface

### [update]
+ fully vectorizedd func with direct masking method
+ numpy func instead of vectorize to run faster
+ random variable
+ object
+ open crack width
+ u_p as a function of w/c
+ random r_v
+ P of f is not consistent between P_RS and crack_condition count. Because the R and S here are dependent: S.max() = R
+ Currently use crack_condition count. for example if 100% fully cracked, then R==S, Pf = 50%, wrong answer


### [TODO]
+ fix negative w_open (surface crack width)

"""


import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np

# model function


def bilinear_stress_strain(epsilon_theta, f_t, E_0):
    """returns the stress in concrete from strain
    epsilon_thetan: numpy array, strain
    f_t : numpy array cracking tensile strength
    E_0 : numpy array modulus of elasticity [TODO: reduction due to creep]

    Notes
    -----
    """
    # parameters defining the bilinear curve
    # critical cracking strain
    epsilon_cr = f_t / E_0
    # strain correspoing to stress = 0.15*f_t
    epsilon_1 = 0.0003
    # strain correspoing to zero residual tensile strength
    epsilon_u = 0.002

    # numpy func for speed, use np.where instead of "if elif else"
    sigma_theta = np.empty_like(epsilon_theta)

    sigma_theta[:] = np.nan

    mask_0 = (epsilon_theta <= epsilon_cr)
    sigma_theta[mask_0] = (E_0 * epsilon_theta)[mask_0]

    mask_1 = ((epsilon_theta > epsilon_cr) & (epsilon_theta <= epsilon_1))
    sigma_theta[mask_1] = (f_t * (1. - 0.85 * (epsilon_theta - epsilon_cr) / (epsilon_1 - epsilon_cr)))[mask_1]

    mask_2 = ((epsilon_theta > epsilon_1) & (epsilon_theta <= epsilon_u))
    sigma_theta[mask_2] = (0.15 * f_t * (epsilon_u - epsilon_theta) / (epsilon_u - epsilon_1))[mask_2]

    # sigma_theta[idx_rest] were set as np.nan

    return sigma_theta


def crack_width_open(a, b, u_st, f_t, E_0):
    """ calculate crack opening on the concret cover surface
    Parameters
    ----------
    a: np array
        inner radius boundary of the rust (center of rebar to rust-concrete) [m]
    b : np array
        outer radius boundary of the concrete (center of rebar to cover surface) [m]
    u_st : array
        rust expansion(to original rebar surface) beyond the porous zone [m]
    f_t : array
        ultimate tensile strength [MPa]
    E_0 : array
        modulus of elasticity [MPa]"""
    epsilon_cr = f_t / E_0
    w = 2 * np.pi * b * (2. / ((b / a)**2. + 1) * u_st / a - epsilon_cr)
    return w


def strain_f(r, a, b, u_st, f_t, E_0, crack_condition):
    """returns the strain along the polar axis r, a<=r<=b
    non-vectorized :single value to single value

    x_loss : array , column vector when passed through strain_stress_crack_f()
        section loss [m]
    everything is array
    r : is matrix with rows representing each r grid, column number is repeated values
    a: array
        inner radius boundary of the rust (center of rebar to rust-concrete) [m]
    b : array
        outer radius boundary of the concrete (center of rebar to cover surface) [m]
    crack_condition: vector array 0 'sound cover', 1 'partially cracked', 2'fully cracked'

    output: epsilon_theta matrix
    """

    # numpy func for speed
    epsilon_theta = np.empty_like(r) * np.nan  # initalize

    # sound cover
    row_mask_0 = (crack_condition == 0)  # 'sound cover':
    elem_mask = ((r >= a) & (r <= b))
    epsilon_theta[row_mask_0 & elem_mask] = (u_st / a * ((b / r)**2 + 1.) / ((b / a)**2 + 1))[row_mask_0 & elem_mask]

    # partially cracked cover
    row_mask_1 = (crack_condition == 1)  # 'partially cracked cover':
    R_c = (b / (f_t * a / (E_0 * u_st) * ((b / a)**2 + 1) - 1)**0.5)
    epsilon_theta[row_mask_1 & elem_mask] = (((b / r)**2 + 1) / ((b / R_c)**2 + 1) * f_t / E_0)[row_mask_1 & elem_mask]

    # fully cracked cover
    row_mask_2 = (crack_condition == 2)  # 'fully cracked cover':
    epsilon_theta[row_mask_2 & elem_mask] = (u_st / a * ((b / r)**2 + 1.) / ((b / a)**2 + 1))[row_mask_2 & elem_mask]

    return epsilon_theta


def strain_stress_crack_f(r, r0_bar, x_loss, cover, f_t, E_0, w_c, r_v, plot=False, ax=None):
    """crack_condition for the whole concrete cover
    r is a matrix
    other scalar like are all array (to be converted to column vector in the calculation)
    """
    epsilon_cr = f_t / E_0
#     r_v = Beta_custom(2.96, 2.96*0.05, 3.3, 2.6)  # volumetric expansion rate  2.96 lower 2.6  upper: 3.3
    u_r = (r_v - 1) * x_loss
    a = r0_bar + u_r  # Rust meets cover, rust-concrete interface
    b = cover + r0_bar
    rust_thickness = u_r - x_loss
    u_p = 12.5e-6 * w_c / 0.5  # m
    u_st = u_r - u_p
    u_st[u_st < 0] = 0

    a = r0_bar + u_r  # Rust meets cover, rust-concrte interface
    b = cover + r0_bar

    a = a[:, None]  # to column vector
    b = b[:, None]
    u_st = u_st[:, None]
    E_0 = E_0[:, None]
    f_t = f_t[:, None]
    epsilon_cr = epsilon_cr[:, None]

    # sound crack
    crack_condition = np.zeros_like(E_0)  # 0: 'sound cover'   initial crack condition guess
    R_c = np.empty_like(E_0) * np.nan
    w_open = np.empty_like(E_0) * np.nan  # crack width on cover

    epsilon_theta = strain_f(r, a, b, u_st, f_t, E_0, crack_condition)

    # partially cracked
    row_mask_1 = ((epsilon_theta >= epsilon_cr).any(axis=1) & (epsilon_theta <= epsilon_cr).any(axis=1))[:, None]

    crack_condition[row_mask_1] = 1  # 1 'partially cracked'
    elem_mask = np.full(r.shape, True, dtype=bool)  # all True

    epsilon_theta[row_mask_1 & elem_mask] = strain_f(r, a, b, u_st, f_t, E_0, crack_condition)[row_mask_1 & elem_mask]

    R_c[row_mask_1] = (b / (f_t * a / (E_0 * u_st) * ((b / a)**2 + 1) - 1)**0.5)[row_mask_1]  # crack tip

    # fully cracked
    # there could be nans in epsilon_theta, compare with Nan is always False
    row_mask_2_inverse = ((epsilon_theta) < epsilon_cr).any(axis=1)[:, None]
    row_mask_2 = ~row_mask_2_inverse
    crack_condition[row_mask_2] = 2  # 2 'fully cracked'
    epsilon_theta[row_mask_2 & elem_mask] = strain_f(r, a, b, u_st, f_t, E_0, crack_condition)[row_mask_2 & elem_mask]
    R_c[row_mask_2] = b[row_mask_2]
    w_open = crack_width_open(a, b, u_st, f_t, E_0)

    # calculate stress
    sigma_theta = bilinear_stress_strain(epsilon_theta, f_t, E_0)

    if plot:
        if ax is None:
            ax = plt.gca()
        ax.plot(r[0, :], epsilon_theta[0, :], color='C0', label=r'strain $\epsilon_{\theta}$')

        # epsilon_1 = 0.0003
        epsilon_u = 0.002
#         ax.hlines(epsilon_cr, r.min(), r.max(),'r', label=r'critical strain $\epsilon_{cr}$')
#         ax.hlines(epsilon_1, r.min(), r.max(),'C1', label=r'$\epsilon_1$')
#         ax.hlines(epsilon_u, r.min(), r.max(), 'b', label=r'zero residual stress strain $\epsilon_u$')
        ax.vlines(a[0], 0, epsilon_u, linestyle='-', label='rust-concrete boundary')
        ax.vlines(R_c[0], 0, epsilon_u, linestyle=':', label='crack tip')

        ax.set_title('crack condition: {}, 0-sound, 1-partial, 2-fully cracked'.format(crack_condition[0]))
        ax.set_xlabel('Radial distance from the center of the rebar,[m]')
        ax.set_ylabel('Strain perpend to radial axis')
        ax.legend()

        ax1 = ax.twinx()
        ax1.plot(r[0, :], sigma_theta[0, :], color='C1', label=r'stress $\sigma_{\theta}$')

        ax1.set_ylabel('Stress perpend to radial axis,[MPa]')
        ax1.legend(loc='center right')
        plt.tight_layout()

        plt.show()
    return epsilon_theta, sigma_theta, rust_thickness, crack_condition, R_c, w_open


def solve_stress_strain_crack_deterministic(pars, number_of_points=100):
    # deterministic with the mean values
    r0_bar_mean = np.array([pars.r0_bar.mean()])
    x_loss_mean = np.array([pars.x_loss.mean()])
    cover_mean = np.array([pars.cover.mean()])
    f_t_mean = np.array([pars.f_t.mean()])
    E_0_mean = np.array([pars.E_0.mean()])
    w_c = np.array([pars.w_c.mean()])
    r_v = np.array([pars.r_v.mean()])

    r = np.full((1, number_of_points), np.linspace(r0_bar_mean[0], cover_mean[0], number_of_points))  # solution point

    fig, ax = plt.subplots()
    epsilon_theta, sigma_theta, rust_thickness, crack_condition, R_c, w_open = strain_stress_crack_f(
        r, r0_bar_mean, x_loss_mean, cover_mean, f_t_mean, E_0_mean, w_c, r_v, plot=True)
    return epsilon_theta, sigma_theta, rust_thickness, crack_condition, R_c, w_open


def solve_stress_strain_crack_stochastic(pars, number_of_points=100):
    # Stochastic with random variables
    r0_bar = pars.r0_bar
    x_loss = pars.x_loss
    cover = pars.cover
    f_t = pars.f_t
    E_0 = pars.E_0
    w_c = pars.w_c
    r_v = pars.r_v

    # every row is a sample of r
    r_mat = np.array([np.linspace(this_r0_bar, this_cover, number_of_points)
                      for (this_r0_bar, this_cover) in zip(r0_bar, cover)])  # solution point

    (epsilon_theta, sigma_theta, rust_thickness, crack_condition, R_c, w_open) = strain_stress_crack_f(
        r_mat, r0_bar, x_loss, cover, f_t, E_0, w_c, r_v)

    return epsilon_theta, sigma_theta, rust_thickness, crack_condition.transpose()[0], R_c.transpose()[0], w_open.transpose()[0]


class Cracking_Model:
    def __init__(self, pars):
        self.pars = pars

    def run(self, stochastic=True):
        if stochastic:
            sol = solve_stress_strain_crack_stochastic(self.pars)  # no plot
        else:
            print('deterministic')
            sol = solve_stress_strain_crack_deterministic(self.pars)  # plot to plt.gca()
            # fig.save()

        self.epsilon_theta, self.sigma_theta, self.rust_thickness, self.crack_condition, self.R_c, self.w_open = sol

    def postproc(self):
        print('for stochastic solution only')
        crack_length_over_cover = (self.R_c - self.pars.r0_bar) / self.pars.cover
        crack_length_over_cover[np.isnan(crack_length_over_cover)] = 0.0  # crack length=0 for no crack
        self.crack_length_over_cover = crack_length_over_cover
        self.crack_visible_rate_count = len(self.crack_condition[self.crack_condition == 2]) / len(self.crack_condition)

    def copy(self):
        return deepcopy(self)
