import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.integrate import dblquad
from scipy import integrate
from scipy.optimize import brentq
from scipy.stats import ttest_1samp
from tqdm.notebook import tqdm
import multiprocessing
from functools import partial
from src.parameters import alpha


def f_star(f, arg):
    """
    Returns a function applied to the argument as an object
    args:
        f: function to apply
        arg: argument 
    returns:
        function f(arg)
    """
    return f(arg)


def multiprocess_N(f, N, tqdm=False):
    """
    Multiprocess a function (f) elementwise over sample sizess (N)
    args:
        f: function to apply
        N: Array of sample sizes (N)
    returns:
        Array of f(N)
    """
    if tqdm == True:
        with multiprocessing.Pool(16) as p:
            out = list(
                tqdm(p.imap(partial(f_star, f), N.ravel()), total=len(N.ravel()))
            )
    else:
        with multiprocessing.Pool(16) as p:
            out = list(p.imap(partial(f_star, f), N.ravel()))

    return np.array(out)


def multiprocess_mesh(f, X, Y):
    """
    Multiprocess a function (f) elementwise over X, Y
    args:
        f: function to apply
        X: 2D array
        Y: 2D array
    returns:
        2D array of f(X, Y)
    """

    def f_star(args):
        return f(*args)

    with multiprocessing.Pool(16) as p:
        out = list(tqdm(p.imap(f, zip(X.ravel(), Y.ravel())), total=len(X.ravel())))
    return np.array(out).reshape(X.shape)


def make_mesh(res, x_low=0, x_high=1, y_low=0, y_high=1):
    """
    Generate a meshgrid of size res x res
    args:
        res: resolution of the meshgrid
        x_low: lower bound of x
        x_high: upper bound of x
        y_low: lower bound of y
        y_high: upper bound of y
    returns:
        X, Y: meshgrid
    """
    x = np.linspace(x_low, x_high, res)
    y = np.linspace(y_low, y_high, res)
    X, Y = np.meshgrid(x, y)
    return X, Y


@np.vectorize
def publication_rate(tau=0.5, sig=0.5, eps=1, n=100, alpha=alpha):
    """
    Compute the publication rate for a given set of parameters.
    params:
        tau: variation in true effect sizes
        sig: standard deviation corresponding to heterogeneity
        eps: error
        n: sample size
        alpha: significance threshold
    returns:
        the publication rate
    """
    publication_rate = 2 * norm.cdf(
        -(eps / np.sqrt(n))
        * norm.ppf(1 - alpha / 2)
        / np.sqrt((eps**2 / n) + tau**2 + sig**2)
    )
    return publication_rate



@np.vectorize
def replicate_rate(tau=0.5, sig=0.5, eps=1, n=100, alpha=alpha):
    """
    Compute the probability of replicating given tau, sigma, epsilon, n and alpha. 
    params:
        tau: variation in true effect sizes
        sig: heterogeneity
        eps: error
        n: sample size
        alpha: significance threshold
    returns:
        the replication rate
    """
    mean = np.array([0, 0])
    sigma = (n / eps**2) * (
        np.reshape(np.repeat(tau**2, 4), (2, 2))
        + np.diag(np.repeat(sig**2 + eps**2 / n, 2))
    )

    dist = mvn(mean=mean, cov=sigma)

    both_positive_rate = dist.cdf([norm.ppf(alpha / 2), norm.ppf(alpha / 2)])
    orig_positive_rate = publication_rate(tau, sig, eps, n) / 2
    return both_positive_rate / orig_positive_rate


@np.vectorize
def sign_error(tau=0.5, sig=0.5, eps=1, n=100, alpha=alpha):
    """
    Compute the probability of sign error given tau, sigma, eps, n and alpha 
    params:
        tau: variation in true effect sizes
        sig: heterogeneity
        eps: error
        n: sample size
        alpha: significance threshold
    returns:
        probability of sign error
    """
    mean = np.array([0, 0])
    sigma = np.matrix(
        [
            [tau**2, (np.sqrt(n) / eps) * tau**2],
            [(np.sqrt(n) / eps) * tau**2, (n / eps**2) * (tau**2 + sig**2) + 1],
        ]
    )

    def integrand(a, b):
        return mvn(mean=mean, cov=sigma, allow_singular=True).pdf([a, b])

    w_pos_z_neg = dblquad(integrand, -np.inf, norm.ppf(alpha / 2), 0, np.inf)[0]

    orig_negative_rate = publication_rate(tau, sig, eps, n) / 2
    return w_pos_z_neg / orig_negative_rate


@np.vectorize
def sign_error2(tau=0.5, sig=0.5, eps=1, n=100, alpha=alpha):
    """
    Alternate conceptualization of sign error examining the probability of a reversal.
    params:
        tau: variation in true effect sizes
        sig: heterogeneity
        eps: error
        n: sample size
        alpha: significance threshold
    returns:
        the probability of a reversal
    """
    mean = np.array([0, 0])
    sigma = np.matrix(
        [
            [tau**2 + sig**2, (np.sqrt(n) / eps) * tau**2],
            [(np.sqrt(n) / eps) * tau**2, (n / eps**2) * (tau**2 + sig**2) + 1],
        ]
    )

    def integrand(a, b):
        return mvn(mean=mean, cov=sigma).pdf([a, b])

    w_pos_z_neg = dblquad(integrand, -np.inf, norm.ppf(alpha / 2), 0, np.inf)[0]

    orig_negative_rate = publication_rate(tau, sig, eps, n) / 2
    return w_pos_z_neg / orig_negative_rate


@np.vectorize
def joint_pdf_uv(u, v, tau, sig, eps, n):
    """
    Helper function for computing the magnitude error.
    params:
        u: the first component of the random vector
        v: the second component of the random vector
        tau: variation in true effect sizes
        sig: heterogeneity
        eps: error
        n: sample size
    returns:
        the joint pdf of u and v
    """
    my_sig_x = tau
    my_sig_y = np.sqrt(sig**2 + eps**2 / n)
    return (
        (1 / (np.pi * my_sig_x * my_sig_y))
        * (v / u**2)
        * (
            np.exp(-((v / u) ** 2) / (2 * my_sig_x**2))
            * (
                np.exp(-((v - v / u) ** 2) / (2 * my_sig_y**2))
                + np.exp(-((v + v / u) ** 2) / (2 * my_sig_y**2))
            )
        )
    )


@np.vectorize
def magnitude_error(tau=0.5, sig=0.5, eps=1, n=100, alpha=alpha):
    """Compute the magnitude error for a given set of parameters.
    params:
        tau: variation in true effect sizes
        sig: heterogeneity
        eps: error
        n: sample size
        alpha: the significance threshold
    returns:
        the magnitude error
    """

    def my_objective(m):
        def integrand(u, v):
            return joint_pdf_uv(u, v, tau, sig, eps, n)

        numerator = dblquad(
            integrand, eps * norm.ppf(1 - alpha / 2) / np.sqrt(n), np.inf, 0, m
        )[0]
        denominator = publication_rate(tau, sig, eps, n)
        return numerator / denominator - 0.5

    res = brentq(my_objective, 1e-20, 1e3)
    return res


def compute_metrics(tau=0.5, sig=0.5, eps=1, n=100, alpha=0.05):
    """Compute four coure metrics using numerical approximation for a fixed set of parameters
    params:
        tau: variation in true effect sizes
        sig: heterogeneity
        eps: error
        n: sample size
        alpha: the significance threshold
    returns:
        a pandas dataframe with numerical results for each metric.  
    """
    my_publication_rate = publication_rate(tau, sig, eps, n, alpha)
    my_replication_rate = replicate_rate(tau, sig, eps, n, alpha)
    my_sign_error = sign_error(tau, sig, eps, n, alpha)
    my_magnitude_error = magnitude_error(tau, sig, eps, n, alpha)
    return pd.DataFrame(
        {
            "type": "numerical",
            "publication_rate": my_publication_rate,
            "replication_rate": my_replication_rate,
            "sign_error": my_sign_error,
            "magnitude_error": my_magnitude_error,
        },
        index=[0],
    )


def simulation(tau=0.5, sig=0.5, eps=1, n=100, n_sim=int(1e4), alpha=alpha):
    """Compute four coure metrics using simulation for a fixed set of parameters
    params:
        tau: variation in true effect sizes
        sig: heterogeneity
        eps: error
        n: sample size
        n_sim: Number of simulations
        alpha: the significance threshold
    returns:
        a pandas dataframe with simulation results for each metric.  
    """
    true_effects = np.random.normal(0, tau, n_sim)
    mediated_effect_orig = np.random.normal(true_effects, sig, n_sim)
    mediated_effect_rep = np.random.normal(true_effects, sig, n_sim)
    data_orig = np.vstack(
        [np.random.normal(item, eps, n) for item in mediated_effect_orig]
    )
    data_rep = np.vstack(
        [np.random.normal(item, eps, n) for item in mediated_effect_rep]
    )
    mean_orig = np.mean(data_orig, axis=1)
    mean_rep = np.mean(data_rep, axis=1)

    z_orig = mean_orig / (eps / np.sqrt(n))
    z_rep = mean_rep / (eps / np.sqrt(n))

    orig_result_pos = z_orig > norm.ppf(1 - alpha / 2)
    orig_result_neg = z_orig < norm.ppf(alpha / 2)
    rep_result_pos = z_rep > norm.ppf(1 - alpha / 2)

    original_result = np.abs(z_orig) > norm.ppf(1 - alpha / 2)
    my_publication_fraction = np.mean(original_result)

    my_replication_fraction = np.sum(rep_result_pos[orig_result_pos]) / np.sum(
        orig_result_pos
    )
    my_sign_error = np.sum((true_effects > 0)[orig_result_neg]) / np.sum(
        orig_result_pos
    )

    my_magnitude_error = np.median(np.abs(mean_orig / true_effects)[original_result])
    return pd.DataFrame(
        {
            "type": "simulation",
            "publication_rate": my_publication_fraction,
            "replication_rate": my_replication_fraction,
            "sign_error": my_sign_error,
            "magnitude_error": my_magnitude_error,
        },
        index=[0],
    )


def run_exp(mu, sigma, n, phack=True):
    """'
    Runs a simulated experiment
    params:
        mu: true average effect size
        sigma: heterogeneity
        n: sample size
        phack: whether the analyst opts to use a p-hacking procedure
    returns:
        sign: the sign of the effect
        t: the t-statistic
        p: the p-value
        effedt_size: the effect size
        correct: whether the effect directin is correctly identified
        significant: whether the effect is significant
        replicate: whether the effect is replicated
    """
    d = np.abs(np.random.normal(0, mu / (np.sqrt(2) / np.sqrt(np.pi))))
    d = np.random.normal(d, sigma)
    x = np.random.normal(d, 1, n)
    test = ttest_1samp(x, 0.0)
    significant = (
        any(ttest_1samp(np.delete(x, i), 0)[1] < 0.05 for i in range(n))
        if phack
        else test[1] < 0.05
    )
    sign = np.sign(d) if significant else 0
    replicate = None
    if significant:
        rep = np.random.normal(d, 1, 200)
        rep_test = ttest_1samp(rep, 0)
        replicate = rep_test[1] < 0.05 and np.sign(d) == sign
    return {
        "P Hacked": "P-Hacked" if phack else "Not P-Hacked",
        "sign": sign,
        "correct": sign == np.sign(mu),
        "effect_size": test[0],
        "pvalue": test[1],
        "significant": significant,
        "replicate": replicate,
    }


from src.parameters import (
    n,
    fig_3_tau,
    fig_2_tau,
    p_hacking_mu,
    p_hacking_sigma,
    p_hacking_n,
)

# We need to do define these functions here so multiprocessing doesn't give us a headache when running
# in a jupyter window
def pub_func(x):
    return publication_rate(tau=x[0], sig=x[1], eps=1, n=n)


def rep_func(x):
    return replicate_rate(tau=x[0], sig=x[1], eps=1, n=n)


def pub_func_n(x):
    return publication_rate(tau=fig_2_tau, sig=x[1], eps=1, n=x[0])


def rep_func_n(x):
    return replicate_rate(tau=fig_2_tau, sig=x[1], eps=1, n=x[0])


def sign_func(x):
    return sign_error(tau=x[0], sig=x[1], eps=1, n=n)


def sign_func2(x):
    return sign_error2(tau=x[0], sig=x[1], eps=1, n=n)


def sign_func2_n(x):
    return sign_error2(tau=fig_3_tau, sig=x[1], eps=1, n=x[0])


def magnitude_func(x):
    return magnitude_error(tau=x[0], sig=x[1], eps=1, n=n)


def sign_func_n(x):
    return sign_error(tau=fig_3_tau, sig=x[1], eps=1, n=x[0])


def magnitude_func_n(x):
    return magnitude_error(tau=fig_3_tau, sig=x[1], eps=1, n=x[0])


def wrapped_run_exp(x):
    return run_exp(mu=p_hacking_mu, sigma=p_hacking_sigma, n=p_hacking_n, phack=x)
