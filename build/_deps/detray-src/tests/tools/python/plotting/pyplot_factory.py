# Detray library, part of the ACTS project (R&D line)
#
# (c) 2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# python includes
from collections import namedtuple
import math
import numpy as np

# python based plotting
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.style as style

style.use("tableau-colorblind10")
# style.use('seaborn-colorblind')

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.size": 25,
        "font.family": "serif",
    }
)


# See: https://stackoverflow.com/questions/42142144/displaying-first-decimal-digit-in-scientific-notation-in-matplotlib
class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self):
        self.format = "%3.1f"


# -------------------------------------------------------------------------------
# Global identifiers
# -------------------------------------------------------------------------------

""" Pass plotting data between functions """
plt_data = namedtuple("plt_data", "fig ax lgd data bins mu rms errors")

""" Wrap the configuration for a legend """
legend_options = namedtuple("legend_options", "loc ncol colspacing handletextpad")

default_color = "tab:blue"

""" Conveniently get the legend options """


def get_legend_options():
    return legend_options("upper right", 1, 1, 1)


# -------------------------------------------------------------------------------
# Data Plotting
# -------------------------------------------------------------------------------

"""
Plotter interface that uses pyplot/matplotlib.
"""


class pyplot_factory:

    def __init__(self, out_dir, logger, atlas_badge=""):
        self.name = ("Pyplot",)
        self.output_prefix = out_dir
        self.logger = logger
        self.atlas_badge = atlas_badge
        self.badge_scale = 1.1
        self.axis_formatter = ScalarFormatterForceFormat()
        self.axis_formatter.set_powerlimits((-2, 2))

    """ Add legend to a plot. Labbels must be defined. """

    def add_legend(self, ax, options=get_legend_options()):
        return ax.legend(
            loc=options.loc,
            ncol=options.ncol,
            borderpad=0.3,
            columnspacing=options.colspacing,
            handletextpad=options.handletextpad,
        )

    """
    Create a histogram from given input data. The normalization is achieved by
    dividing the bin count by the total number of observations. The error is
    calculated as the square root of the bin content.
    """

    def hist1D(
        self,
        x,
        errors=None,
        w=None,
        x_label="x",
        y_label="",
        title="",
        label="",
        x_min=None,
        x_max=None,
        bins=1,
        color=default_color,
        alpha=0.75,
        set_log=False,
        normalize=False,
        show_error=False,
        show_stats=True,
        u_outlier=-1,
        o_outlier=-1,
        figsize=(8, 8),
        lgd_ops=get_legend_options(),
        layout="constrained",
        ax_formatter=None,
    ):

        # Create fresh plot
        fig = plt.figure(figsize=figsize, layout=layout)
        ax = fig.add_subplot(1, 1, 1)

        if ax_formatter is None:
            ax.xaxis.set_major_formatter(self.axis_formatter)
            ax.yaxis.set_major_formatter(self.axis_formatter)

        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))

        # Do calculations on data in the range of the histogram
        if x_min is not None and x_max is not None:
            x = x[np.nonzero(x >= x_min)]
            x = x[np.nonzero(x <= x_max)]
        else:
            x_min = np.min(x)
            x_max = np.max(x)

        # Display number of entries in under- and overflow bins
        underflow = len(np.argwhere(x < x_min))
        overflow = len(np.argwhere(x > x_max))
        if u_outlier >= 0 or o_outlier >= 0:
            underflow = underflow + u_outlier
            overflow = overflow + o_outlier

        # Nothing left to do
        if len(x) == 0:
            self.logger.debug(rf" create hist: empty data {label}")
            return plt_data(fig, ax, None, None, None, None, None, None)

        # Histogram normalization
        scale = 1.0 / len(x) if normalize else 1.0

        # Format the 'newline'
        newline = "\n"

        # Name of the datat collection
        label_str = f"{label} ({len(x)} entries)"
        if u_outlier >= 0 or o_outlier >= 0:
            label_str = (
                label_str
                + f"{newline} underflow: {underflow}"
                + f"{newline} overflow:  {overflow}"
            )

        # Fill data
        data, bins, _ = ax.hist(
            x,
            weights=w,
            range=(x_min, x_max),
            bins=bins,
            label=label_str,
            histtype="stepfilled",
            density=normalize,
            facecolor=mcolors.to_rgba(color, alpha),
            edgecolor=color,
        )

        # Add some additional information
        if show_stats:
            mean = np.mean(x, axis=0)
            # rms  = np.sqrt(np.mean(np.square(x)))
            stdev = np.std(x, axis=0)

            # Create empty plot with blank marker containing the extra label
            ax.plot(
                [],
                [],
                " ",
                label=rf"data:"
                rf"{newline}mean    = {mean:.2e}"
                rf"{newline}stddev  = {stdev:.2e}",
            )
        else:
            mean = None
            stdev = None

        # Refine plot
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25)

        # Add legend
        lgd = self.add_legend(ax, lgd_ops)

        # Adjust spacing in box
        lgd.legend_handles[0].set_visible(False)
        if show_stats:
            lgd.legend_handles[1].set_visible(False)
        for vpack in lgd._legend_handle_box.get_children():
            for hpack in vpack.get_children():
                hpack.get_children()[0].set_width(0)

        # Calculate the bin error
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        err = np.sqrt(scale * data) if errors is None else errors
        if show_error or errors is not None:
            ax.errorbar(
                bin_centers,
                data,
                yerr=err,
                fmt=".",
                linestyle="",
                linewidth=0.4,
                color="black",
                capsize=2.5,
            )

        # Plot log scale
        if set_log:
            ax.set_yscale("log")

        return plt_data(fig, ax, lgd, data, bins, mean, stdev, err)

    """ Add new data to an existing plot """

    def add_plot(
        self,
        old_hist,
        x,
        errors=None,
        w=None,
        label="",
        color="tab:orange",
        alpha=0.75,
        normalize=False,
        show_error=False,
    ):

        # do calculations on data in the range of the histogram
        x_min = np.min(old_hist.bins)
        x_max = np.max(old_hist.bins)

        x = x[np.nonzero(x >= x_min)]
        x = x[np.nonzero(x <= x_max)]

        # Nothing left to do
        if len(x) == 0 or old_hist.data is None:
            self.logger.debug(rf" add hist: empty data {label}")
            return old_hist

        # Add new data to old hist axis
        scale = 1.0 / len(x) if normalize else 1.0
        data, bins, _ = old_hist.ax.hist(
            x=x,
            bins=old_hist.bins,
            label=f"{label} ({len(x)} entries)",
            weights=w,
            histtype="stepfilled",
            facecolor=mcolors.to_rgba(color, alpha),
            edgecolor=color,
        )

        # Update legend
        lgd = old_hist.lgd
        handles, labels = lgd.axes.get_legend_handles_labels()
        lgd._legend_box = None
        lgd._init_legend_box(handles, labels)
        lgd._set_loc(lgd._loc)
        lgd.set_title(lgd.get_title().get_text())

        # Calculate the bin error
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        err = np.sqrt(scale * data) if errors is None else errors
        if show_error or errors is not None:
            old_hist.ax.errorbar(
                bin_centers,
                data,
                yerr=err,
                fmt=".",
                linestyle="",
                linewidth=0.4,
                color="black",
                capsize=2.5,
            )

        return plt_data(
            old_hist.fig, old_hist.ax, old_hist.lgd, data, bins, None, None, err
        )

    """
    Plot the ratio of two histograms. The data is assumed to be uncorrelated.
    """

    def add_ratio(
        self, nom, denom, label, color="tab:red", set_log=False, show_error=False
    ):

        # Resize figure
        nom.fig.set_figheight(7)
        nom.fig.set_figwidth(8)

        if nom.bins is None or denom.bins is None:
            return plt_data(nom.fig, nom.ax, None, None, None, None, None, None)

        if len(nom.bins) != len(denom.bins):
            return plt_data(nom.fig, nom.ax, None, None, None, None, None, None)

        # Remove ticks/labels that are already visible on the ratio plot
        old_x_label = nom.ax.xaxis.get_label().get_text()
        nom.ax.tick_params(
            axis="x", which="both", bottom=True, top=False, labelbottom=False
        )
        nom.ax.set_xlabel("")

        # Don't print a warning when dividing by zero
        with np.errstate(divide="ignore"), np.errstate(invalid="ignore"):
            # Filter out nan results from division by zero
            ratio = np.nan_to_num(nom.data / denom.data, nan=0, posinf=0)

            # Calculate errors by Gaussian propagation
            bin_centers = 0.5 * (nom.bins[1:] + nom.bins[:-1])
            n_data, d_data = (nom.data, denom.data)

            # Gaussian approximation for large number of events in bin
            # Note: Should be Clopper-Pearson
            n_err, d_err = (nom.errors, denom.errors)
            errors = np.nan_to_num(
                np.sqrt(
                    np.square(n_err / d_data)
                    + np.square(n_data * d_err / np.square(d_data))
                ),
                nan=0,
                posinf=0,
            )

        # create new axes on the bottom of the current axes
        # The first argument of the new_vertical(new_horizontal) method is
        # the height (width) of the axes to be created in inches.
        divider = make_axes_locatable(nom.ax)
        ratio_plot = divider.append_axes("bottom", 1.2, pad=0.2, sharex=nom.ax)
        if show_error:
            ratio_plot.errorbar(
                bin_centers, ratio, yerr=errors, label=label, color=color, fmt="."
            )
        else:
            ratio_plot.plot(
                bin_centers,
                ratio,
                label=label,
                color=color,
                marker=".",
                linestyle="",
            )
        # Refine plot
        ratio_plot.set_xlabel(old_x_label)
        ratio_plot.set_ylabel("ratio")
        ratio_plot.grid(True, alpha=0.25)
        # Plot log scale
        if set_log:
            ratio_plot.set_yscale("log")

        # Add a horizontal blue line at y = 1.
        ratio_plot.axline((nom.bins[0], 1), (nom.bins[-1], 1), linewidth=1, color="b")

        # Add legend
        # lgd = self.add_legend(ratio_plot)
        # Move the legend
        # lgd.set_bbox_to_anchor((box_anchor_x, box_anchor_y))
        nom.fig.set_size_inches((9, 9))

        return plt_data(nom.fig, ratio_plot, None, None, None, None, None, errors)

    """
    Create a 2D histogram from given input data. If z values are given they will
    be used as weights per bin.
    """

    def hist2D(
        self,
        x,
        y,
        z=None,
        x_label="x",
        y_label="y",
        z_label="",
        title="",
        label="",
        x_min=None,
        x_max=None,
        x_bins=1,
        y_min=None,
        y_max=None,
        y_bins=1,
        color=default_color,
        alpha=0.75,
        show_stats=True,
        figsize=(8, 6),
    ):

        # Create fresh plot
        fig = plt.figure(figsize=figsize, layout="constrained")
        ax = fig.add_subplot(1, 1, 1)

        # Do calculations on data in the range of the histogram
        if x_min is not None and x_max is not None:
            x = x[np.nonzero(x >= x_min)]
            x = x[np.nonzero(x <= x_max)]
        else:
            x_min = np.min(x)
            x_max = np.max(x)

        if y_min is not None and y_max is not None:
            y = y[np.nonzero(y >= y_min)]
            y = y[np.nonzero(y <= y_max)]
        else:
            y_min = np.min(y)
            y_max = np.max(y)

        # Nothing left to do
        if len(x) == 0 or len(y) == 0:
            self.logger.debug(rf" create hist: empty data {label}")
            return plt_data(fig, ax, None, None, None, None, None, None)

        # Fill data
        data, _, _, hist = ax.hist2d(
            x,
            y,
            weights=z,
            range=[(x_min, x_max), (y_min, y_max)],
            bins=(x_bins, y_bins),
            label=f"{label}  ({len(x)*len(y)} entries)",
            facecolor=mcolors.to_rgba(color, alpha),
            edgecolor=None,
            rasterized=True,
        )

        # Add some additional information
        if show_stats:
            x_mean = np.mean(x, axis=0)
            x_rms = np.sqrt(np.mean(np.square(x)))
            y_mean = np.mean(y, axis=0)
            y_rms = np.sqrt(np.mean(np.square(y)))

            # Create empty plot with blank marker containing the extra label
            newline = "\n"
            ax.plot(
                [],
                [],
                " ",
                label=rf"xMean = {x_mean:.2e}"
                rf"{newline}xRMS  = {x_rms:.2e}"
                rf"yMean = {y_mean:.2e}"
                rf"{newline}yRMS  = {y_rms:.2e}",
            )

        # Refine plot
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Add the colorbar
        fig.colorbar(hist, label=z_label)

        return plt_data(fig, ax, None, data, None, None, None, None)

    """ Create a 2D scatter plot """

    def scatter(
        self,
        x,
        y,
        x_label="",
        y_label="",
        title="",
        label="",
        color=default_color,
        alpha=1,
        figsize=(8, 6),
        show_stats=lambda x, _: f"{len(x)} entries",
        lgd_ops=get_legend_options(),
    ):

        fig = plt.figure(figsize=figsize, layout="constrained")
        ax = fig.add_subplot(1, 1, 1)

        # Create empty plot with blank marker containing the extra label
        ax.plot([], [], " ", label=show_stats(x, y))
        scatter = ax.scatter(
            x, y, label=label, c=color, s=0.1, alpha=alpha, rasterized=True
        )

        # Refine plot
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25)

        # Add legend
        lgd = self.add_legend(ax, lgd_ops)

        # Refine legend
        lgd.legend_handles[0].set_visible(False)
        for handle in lgd.legend_handles[1:]:
            handle.set_sizes([40])

        # Adjust spacing in box
        for vpack in lgd._legend_handle_box.get_children()[:1]:
            for hpack in vpack.get_children():
                hpack.get_children()[0].set_width(0)

        return plt_data(fig, ax, lgd, scatter, None, None, None, None)

    """ Add new data in a different color to a scatter plot """

    def highlight_region(self, plot_data, x, y, color, label=""):

        if label == "":
            plot_data.ax.scatter(x, y, c=color, alpha=1, s=0.1, rasterized=True)
        else:
            plot_data.ax.scatter(
                x, y, c=color, alpha=1, s=0.1, label=label, rasterized=True
            )

            # Update legend
            lgd = plot_data.lgd
            handles, labels = lgd.axes.get_legend_handles_labels()
            lgd._legend_box = None
            lgd._init_legend_box(handles, labels)
            lgd._set_loc(lgd._loc)
            lgd.set_title(lgd.get_title().get_text())

        # Refine legend
        lgd.legend_handles[0].set_visible(False)
        for handle in lgd.legend_handles[1:]:
            handle.set_sizes([40])

        # Adjust spacing in box
        for vpack in lgd._legend_handle_box.get_children()[:1]:
            for hpack in vpack.get_children():
                hpack.get_children()[0].set_width(0)

    """ Fit a Gaussian to a 1D distribution and plot in the same figure. """

    def fit_gaussian(self, dist):

        # Calculate bin centers from bin edges
        bins = dist.bins
        if bins is None:
            # If fit failed, return empty result
            return None, None

        bin_centers = [(b1 + b2) / 2 for b1, b2 in zip(bins, bins[1:])]

        # Gaussian distribution with all fit parameters
        def gaussian(x, a, mean, sigma):
            return (
                a
                / (math.sqrt(2 * math.pi) * sigma)
                * np.exp(-((x - mean) ** 2 / (2 * sigma**2)))
            )

        # Gaussian fit
        try:
            from scipy.optimize import curve_fit
        except ImportError:
            print("WARNING: Could not find scipy: Skipping fit")
        else:
            try:
                # Initial estimators
                mean = np.mean(bin_centers, axis=0)
                sigma = np.std(bin_centers, axis=0)
                a = np.max(dist.data) * (math.sqrt(2 * math.pi) * sigma)

                popt, _ = curve_fit(
                    gaussian, bin_centers, dist.data, p0=[a, mean, sigma]
                )
            except RuntimeError:
                # If fit failed, return empty result
                return None, None

            # If the fitting was successful, plot the curve
            mu = float(f"{popt[1]:.2e}")  # < formatting the sig. digits
            sig = float(f"{popt[2]:.2e}")
            newline = "\n"

            # Generate points for the curve
            min_val = min(bin_centers)
            max_val = max(bin_centers)
            step = (max_val - min_val) / 1000
            x = [v for v in np.arange(min_val, max_val + step, step)]

            dist.ax.plot(
                x,
                gaussian(x, *popt),
                label=rf"gaussian fit:{newline}$\mu$ = {mu:.2e}"
                + rf"{newline}$\sigma$ = {abs(sig):.2e}",
                color="tab:orange",
            )

            # Update legend
            lgd = dist.lgd
            handles, labels = lgd.axes.get_legend_handles_labels()
            lgd._legend_box = None
            lgd._init_legend_box(handles, labels)
            lgd._set_loc(lgd._loc)
            lgd.set_title(lgd.get_title().get_text())

            # Adjust spacing in box
            lgd.legend_handles[0].set_visible(False)
            for vpack in lgd._legend_handle_box.get_children()[:-1]:
                for hpack in vpack.get_children():
                    hpack.get_children()[0].set_width(0)

            return popt[1], abs(popt[2])

        return None, None

    """ Safe a plot to disk """

    def write_plot(
        self, plot_data, name="plot", file_format="svg", out_prefix="", dpi=450
    ):
        if out_prefix == "":
            file_name = self.output_prefix + name + "." + file_format
        else:
            file_name = out_prefix + name + "." + file_format

        plot_data.fig.savefig(file_name, dpi=dpi)
        plt.close(plot_data.fig)

    """ Safe a plot as svg """

    def write_svg(self, plot_data, name, out_prefix=""):

        self.write_plot(plot_data, name, ".svg", out_prefix)

    """ Safe a plot as pdf """

    def write_pdf(self, plot_data, name, out_prefix=""):

        self.write_plot(plot_data, name, ".pdf", out_prefix)

    """ Safe a plot as png """

    def write_png(self, plot_data, name, out_prefix=""):

        self.write_plot(plot_data, name, ".png", out_prefix)
