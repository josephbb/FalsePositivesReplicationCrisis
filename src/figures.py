import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import string
import matplotlib as mpl
from src.parameters import *
from src.theory_sims import *


def label_subfigs(axs):
    """Labels subfigs A, B, C, D, etc..."""
    for n, ax in enumerate(axs.flatten()):
        ax.text(
            -0.1,
            1.1,
            string.ascii_uppercase[n],
            transform=ax.transAxes,
            size=20,
            weight="bold",
        )


def plot_contour(
    axs,
    data_dict,
    llmin=0.2,
    llmax=1,
    llstep=9,
    cmap=pub_cmap,
    levelmin=0,
    levelmax=1,
    levelstep=1000,
    log=False,
    llshardcode=False,
    xlabel="",
    ylabel="",
    title="",
):
    """
    Plot a contour plot for our theory figures.
    Args:
        axs: axes object
        data_dict: dictionary of data containing X, Y, Z as 2-D arrays
        llmin: minimum value for contour lines
        llmax: maximum value for contour lines
        llstep: number of contour lines
        cmap: color map
        levelmin: minimum value for contour fill
        levelmax: maximum value for contour fill
        levelstep: number of contour fill levels
        log: whether to use a log scale
        llshardcode: hardcode contour lines
        xlabel: x axis label
        ylabel: y axis label
        title: title
    Returns:
        CS: contour lines
        CSF: contour fill
        levels: contour fill levels
        lls: contour line levels
    """
    X = data_dict["X"]
    Y = data_dict["Y"]
    Z = data_dict["Z"]

    if log:
        lls = np.logspace(llmin, llmax, llstep, base=2)
        levels = np.logspace(levelmin, levelmax, levelstep, base=2)

    else:
        lls = np.linspace(llmin, llmax, llstep)
        levels = np.linspace(levelmin, levelmax, levelstep)

    if llshardcode is not False:
        lls = llshardcode

    CS = axs.contour(X, Y, Z, levels=lls, zorder=2, colors="white")
    axs.clabel(CS, colors="w", fontsize=12)  # contour line labels
    if log:
        CSF = axs.contourf(
            X,
            Y,
            Z,
            zorder=1,
            levels=levels,
            cmap=sns.color_palette(cmap, as_cmap=True),
            norm=mpl.colors.LogNorm(vmin=0.5, vmax=16),
        )
    else:
        CSF = axs.contourf(
            X, Y, Z, zorder=1, levels=levels, cmap=sns.color_palette(cmap, as_cmap=True)
        )

    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    axs.set_title(title, fontsize=16)

    return CS, CSF, levels, lls


def figure_creator(xplots=2, yplots=2):
    """
    Create a 2x2 figure with the same style
    Args:
        xplots: number of rowsq
        yplots: number of columns
    Returns:
        fig, axs: figure and axes objects
    """
    sns.set_context("paper", font_scale=font_scale)
    sns.set_palette(sns.color_palette(pub_cmap))

    fig, axs = plt.subplots(xplots, yplots, figsize=(4 * yplots, 4 * xplots))
    sns.set_context("paper")
    return fig, axs


def fig_1(save_loc="./output/figures/Figure1.png"):
    """
    Figure 1: Overview of theory
    args:
        save_loc: location to save figure
    returns:
            fig: figure object
            ax1: axes object
    """
    pal = sns.color_palette("colorblind", n_colors=6)
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("white")

    fig = plt.figure(figsize=(8, 4))
    gs = fig.add_gridspec(1, 1)
    ax1 = fig.add_subplot(gs[0, :])
    sns.set_style("white")

    ax1.fill_between(
        np.linspace(0, 2, 100),
        np.zeros(100),
        stats.norm(0, 0.2).pdf(np.linspace(0, 2, 100)),
        color=pal[0],
        alpha=0.5,
        zorder=1,
    )
    ax1.fill_between(
        np.linspace(0, 2, 100),
        np.zeros(100),
        stats.norm(loc=0.2, scale=0.1).pdf(np.linspace(0, 2, 100)) / 2,
        color=pal[1],
        alpha=0.3,
        zorder=3,
    )

    ax1.fill_between(
        np.linspace(0, 2, 100),
        np.zeros(100),
        stats.norm(loc=0.3, scale=0.03).pdf(np.linspace(0, 2, 100)) / 20,
        color=pal[2],
        alpha=0.3,
        zorder=4,
    )

    # Label Distributions
    ax1.plot([0, 0.2], [1.2, 1.2], color="k")
    ax1.text(x=0.1, y=1.25, s=r"$\tau$", color="k")

    ax1.plot([0.2, 0.2], [0, 2], color="k", ls="--")
    ax1.text(x=0.28, y=1.25, s=r"$\sigma$", color="k")

    ax1.text(x=0.31, y=0.1, s=r"$\frac{\epsilon}{\sqrt{n}}$")
    ax1.plot([0.3, 0.34], [0.27, 0.27], color="k")

    # Add line for d_true
    ax1.plot([0.2, 0.31], [1.1, 1.1], color="k")
    ax1.text(x=0.2, y=2.1, s=r"$d_{j}$")

    # Add line for original effect size
    ax1.plot(
        [0.1, 0.1],
        [0, stats.norm(loc=0.25, scale=0.1).pdf(0.1) / 2],
        ls="--",
        color="k",
        zorder=3,
    )
    ax1.text(x=0.25, y=0.73, s=r"$d_{original}$")

    # Add line for replication effectsize and label
    ax1.plot(
        [0.3, 0.3],
        [0, stats.norm(loc=0.25, scale=0.1).pdf(0.4) / 2],
        ls="--",
        color="k",
    )
    ax1.text(x=0.05, y=0.73, s=r"$d_{replication}$")

    # Add line for t_crit and label
    ax1.plot([0.22, 0.22], [0, 0.4], color="k")
    ax1.text(x=0.21, y=0.43, s=r"$z_{crit.}$")

    ax1.set_xticks(np.linspace(0, 1, 11))
    ax1.set_xlim(0, 0.8)
    ax1.set_ylim(0, 2.3)
    ax1.set_xlabel("Effect size")
    ax1.set_ylabel("Density")
    plt.tight_layout()

    plt.savefig(save_loc, dpi=300, transparent=False)
    return fig, ax1


def fig_2(analytical_data_fig2, output="./output/figures/figure2.png"):
    """
    Figure 2: Contour plots of the probability of publication, replication
    Args:
        analytical_data_fig2 (dict): Dictionary containing the data for the contour plots
        output (str): Path to save the figure
    Returns:
        fig (matplotlib.figure.Figure): Figure object
        axs (matplotlib.axes._subplots.AxesSubplot): Axes object
    """
    fig, axs = figure_creator()

    _ = plot_contour(
        axs[0][0],
        analytical_data_fig2["2A"],
        xlabel=tau_label,
        ylabel=sigma_label,
        title=publish_title,
    )

    _ = plot_contour(
        axs[0][1],
        analytical_data_fig2["2B"],
        cmap=rep_cmap,
        xlabel=tau_label,
        ylabel=sigma_label,
        title=rep_title,
    )

    _ = plot_contour(
        axs[1][0],
        analytical_data_fig2["2C"],
        cmap=pub_cmap,
        xlabel=sample_size_label,
        ylabel=sigma_label,
        title=publish_title,
    )

    _ = plot_contour(
        axs[1][1],
        analytical_data_fig2["2D"],
        cmap=rep_cmap,
        xlabel=sample_size_label,
        ylabel=sigma_label,
        title=rep_title,
    )

    label_subfigs(axs)
    # Fix layout
    plt.tight_layout()
    # Save figure, return
    plt.savefig(output, dpi=300)
    return fig, axs


def label_subfigs(axs):
    for n, ax in enumerate(axs.flatten()):
        ax.text(
            -0.1,
            1.1,
            string.ascii_uppercase[n],
            transform=ax.transAxes,
            size=20,
            weight="bold",
        )


def fig_3(analytical_data_fig3, output="./output/figures/figure3.png"):
    """
    Figure 3: Contour plots of the probability of publication, replication
    Args:
        analytical_data_fig3 (dict): Dictionary containing the data for the contour plots
        output (str): Path to save the figure
    Returns:
        fig : Figure object
        axs : Axes object
    """

    fig, axs = figure_creator()

    _ = plot_contour(
        axs[0][0],
        analytical_data_fig3["3A"],
        cmap=sign_error_cmap,
        llmin=0,
        llmax=1,
        llstep=11,
        xlabel=tau_label,
        ylabel=sigma_label,
        title=sign_title,
    )

    _ = plot_contour(
        axs[0][1],
        analytical_data_fig3["3B"],
        cmap=magnitude_error_cmap,
        levelmax=8,
        llmax=4,
        llstep=5,
        llmin=0,
        log=True,
        xlabel=tau_label,
        ylabel=sigma_label,
        title=magnitude_title,
    )

    _ = plot_contour(
        axs[1][0],
        analytical_data_fig3["3C"],
        cmap=sign_error_cmap,
        llmin=0,
        llmax=1,
        llstep=11,
        xlabel=sample_size_label,
        ylabel=sigma_label,
        title=sign_title,
    )

    _ = plot_contour(
        axs[1][1],
        analytical_data_fig3["3D"],
        cmap=magnitude_error_cmap,
        levelmax=8,
        log=True,
        llshardcode=np.array([1, 1.5, 2, 3, 4, 5, 6]),
        xlabel=sample_size_label,
        ylabel=sigma_label,
        title=magnitude_title,
    )

    label_subfigs(axs)

    # Fix layout
    plt.tight_layout()
    # Save figure, return
    plt.savefig(output, dpi=300)

    return fig, axs


def si_1(analytical_data_supp1, save_loc="./output/figures/si_1.png"):
    """

    Args:
        analytical_data_fig3 (dict): Dictionary containing the data for the contour plots
        output (str): Path to save the figure
    Returns:
        fig : Figure object
        axs : Axes object
    """
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    sns.set_context("paper", font_scale=1.25)
    sns.set_style("white")

    _ = plot_contour(
        axs[0],
        analytical_data_supp1["S1A"],
        xlabel=r"$\tau$" + " (effect size)",
        ylabel=r"$\sigma$" + " (varying effects)",
        title=sign_title,
        cmap=sign_error_cmap,
        llshardcode=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
    )

    _ = plot_contour(
        axs[1],
        analytical_data_supp1["S1B"],
        xlabel=r"$\tau$" + " (effect size)",
        ylabel=r"$\sigma$" + " (varying effects)",
        cmap=pub_cmap,
        title="Pr(publish)",
    )

    _ = plot_contour(
        axs[2],
        analytical_data_supp1["S1C"],
        xlabel=r"$\tau$" + " (effect size)",
        ylabel=r"$\sigma$" + " (varying effects)",
        cmap="viridis",
        title="Pr(Publish, Correct Sign)",
    )

    for ax in axs.flatten():
        ax.annotate(
            xy=(0.10, 0.8),
            text="",
            xytext=(0.10, -0.01),
            color="k",
            arrowprops=dict(arrowstyle="->", color="k"),
        )

    label_subfigs(axs)
    plt.tight_layout()
    plt.savefig(save_loc, dpi=300)


def fig_4(steps=2):
    """
    Plot Figure 4
    Args:
        steps (int): Number of steps evaluate on the x-axis
    Returns:
        fig (matplotlib.figure.Figure): Figure object
    """
    fig, axs = figure_creator()

    x = np.linspace(0, 1, steps)
    pal1 = sns.color_palette("mako", n_colors=5)

    plt.sca(axs[0][0])
    for idx, sigma in enumerate([0.8, 0.6, 0.4, 0.2, 0.1]):
        y = x * publication_rate(tau=0.4, sig=sigma, n=50) + (1 - x) * sign_error(
            tau=0.4, sig=sigma, n=50
        )
        plt.plot(x, y, color=pal1[idx], label=r"$\sigma=" + str(sigma) + "$")
        plt.plot(
            x,
            np.ones(x.size) * publication_rate(tau=0.4, sig=sigma, n=50),
            ls="--",
            color=pal1[idx],
        )
    plt.legend()
    plt.ylabel(publish_title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Selective \nReporting")
    plt.xlabel("P(W>0)")

    pal2 = sns.color_palette("ch:start=1.3,rot=-.1", n_colors=5)

    plt.sca(axs[0][1])
    x = np.linspace(0, 1, steps)
    for idx, sigma in enumerate([0.8, 0.6, 0.4, 0.2, 0.1]):
        type_s = sign_error(tau=0.4, sig=sigma, n=50)
        publish = publication_rate(tau=0.4, sig=sigma, n=50)
        plt.plot(
            x,
            (1 - x) * type_s / ((1 - x) * type_s + (1 - type_s) * x),
            color=pal2[idx],
            label=r"$\sigma=" + str(sigma) + "$",
        )
        plt.plot(x, np.ones(x.shape) * type_s, color=pal2[idx], ls="--")
    plt.legend()
    plt.ylabel("P(Sign Error)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Selective \nReporting")
    plt.xlabel("P(W>0)")

    plt.sca(axs[1][0])
    palg = sns.color_palette("Greys", n_colors=5)
    x = np.round(np.logspace(1, 3, steps)).astype("int")
    for idx, alpha in enumerate([0.01, 0.05, 0.1, 0.2]):
        y = sign_error(tau=0.4, sig=0.4, n=x, alpha=alpha)
        plt.plot(x, y, color=palg[idx], label=r"$\alpha=" + str(alpha) + "$")
    plt.legend()
    plt.xlim(10, 1000)

    plt.title("Post-hoc " + r"$\alpha$")

    plt.ylabel(sign_title)
    plt.ylim(0, 1)
    plt.xlabel(sample_size_label)

    plt.sca(axs[1][1])
    x = np.round(np.logspace(1, 3, steps)).astype("int")
    for idx, alpha in enumerate([0.01, 0.05, 0.1, 0.2]):
        y = magnitude_error(tau=0.4, sig=0.4, n=x, alpha=alpha)
        plt.plot(x, y, color=palg[idx], label=r"$\alpha=" + str(alpha) + "$")
    plt.legend()
    plt.xlim(10, 1000)
    plt.ylim(0, 3.5)
    plt.title("Post-hoc " + r"$\alpha$")
    plt.ylabel(magnitude_title)
    plt.xlabel(sample_size_label)

    import string

    axs = axs.flat
    for n, ax in enumerate(axs):
        ax.text(
            -0.1,
            1.1,
            string.ascii_uppercase[n],
            transform=ax.transAxes,
            size=20,
            weight="bold",
        )

    plt.tight_layout()
    plt.savefig("./output/figures/fig_4.png", dpi=300, bbox_inches="tight")

    return plt.gcf(), plt.gca()


def si_2(prop_correct, save_loc="./output/figures/si_2.png"):
    # Create the bar plot
    sns.set_palette(["#999999", "#333333"])
    sns.barplot(x="P Hacked", y="proportion", hue="correct", data=prop_correct)

    # Add axis labels and a title
    plt.xlabel("P Hacked")
    plt.ylabel("Proportion of tested hypotheses")

    # Change the legend label
    handles, labels = plt.gca().get_legend_handles_labels()
    labels = ["incorrect", "correct"]
    plt.legend(handles, labels, title="Direction of effect")
    plt.ylim(0, 0.5)
    # Show the plot
    plt.tight_layout()
    plt.savefig(save_loc, dpi=300)

    return plt.gcf(), plt.gca()


def fig_alt_sign(
    alternate_sign_data, output="./output/figures/alternate_sign_data.png"
):
    """
    Figure alt_sign: Contour plots of alternate sign error
        alternate_sign_data (dict): Dictionary containing the data for the contour plots
        output (str): Path to save the figure
    Returns:
        fig (matplotlib.figure.Figure): Figure object
        axs (matplotlib.axes._subplots.AxesSubplot): Axes object
    """

    fig, axs = figure_creator(1, 2)

    _ = plot_contour(
        axs[0],
        alternate_sign_data["S2A"],
        cmap=sign_error_cmap,
        llmin=0,
        llmax=1,
        llstep=11,
        xlabel=tau_label,
        ylabel=sigma_label,
        title=sign_title,
    )

    _ = plot_contour(
        axs[1],
        alternate_sign_data["S2B"],
        cmap=sign_error_cmap,
        llmin=0,
        llmax=1,
        llstep=11,
        xlabel=sample_size_label,
        ylabel=sigma_label,
        title=sign_title,
    )

    label_subfigs(axs)

    # Fix layout
    plt.tight_layout()
    # Save figure, return
    plt.savefig(output, dpi=300)

    return fig, axs


def plot_prior_ppd(values, bins, ax):

    out = []
    for idx in range(values.shape[0]):
        out.append(np.histogram(values[idx], bins=bins)[0])

    out = np.vstack([item / np.sum(item) for item in out])
    y = np.mean(out, axis=0)

    for idx in range(bins.shape[0] - 1):
        ax.plot([bins[idx], bins[idx + 1]], [y[idx], y[idx]], lw=1, color="darkred")

    for percentile in [3, 7, 11, 25, 50]:
        low = np.percentile(out, percentile, axis=0)
        high = np.percentile(out, 100 - percentile, axis=0)
        for idx in range(bins.shape[0] - 1):

            ax.fill_between(
                [bins[idx], bins[idx + 1]],
                [low[idx], low[idx]],
                [high[idx], high[idx]],
                alpha=0.2,
                color="darkred",
                lw=0,
            )


def rank_plot(temp, ax, bins=10):
    plt.sca(ax)
    sns.histplot(temp["rank"], color="darkred", bins=np.linspace(0, 4000, bins))
    expected = temp["rank"].shape[0] / (bins - 1)
    low = stats.binom(temp["rank"].shape[0], 1 / (bins - 1)).ppf(0.03)
    high = stats.binom(temp["rank"].shape[0], 1 / (bins - 1)).ppf(0.97)
    plt.fill_between([0, 4000], [low, low], [high, high], alpha=0.4, color="grey")
    plt.plot([0, 4000], [expected, expected], "k--")
    plt.xlim(0, 4000)
    plt.ylim(0, expected * 1.5)


def eye_test(temp, ax):
    plt.sca(ax)
    plt.scatter(1 - temp["contraction"], temp["z_score"], alpha=0.3, color="darkred")
    plt.xlim(0, 1)
    plt.xlabel("Contraction")
    plt.ylabel("Z-Score")
    plt.xlim(0, 1)
    plt.ylim(-5, 5)
