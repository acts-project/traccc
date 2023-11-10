#!/usr/bin/env python3


import numpy
import pandas
import matplotlib.pyplot
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import pathlib
import typing


BinSpec = (typing.Optional[float], typing.Optional[float], typing.Optional[int])


def range_type(s: str) -> BinSpec:
    v = s.split(":")

    lo = float(v[0]) if v[0] else None
    hi = float(v[1]) if v[1] else None
    bn = int(v[2]) if len(v) == 3 else None

    return (lo, hi, bn)


def pretty_print(col):
    if col == "eta":
        return "$\eta$"
    elif col == "phi":
        return "$\phi$"
    elif col == "pt":
        return "$p_T$"
    else:
        return col


def make_plot(pdf, q, col, bins, logx=False):
    r = (
        q.groupby(["event_id", pandas.cut(q[col], bins)])
        .agg(
            {
                "seed_count": [
                    ("total_tracks", "count"),
                    ("matched_tracks", lambda x: len(x[x > 0.6])),
                    ("total_seeds", "sum"),
                ]
            }
        )
        .droplevel(axis=1, level=0)
    )
    r["efficiency"] = r["matched_tracks"] / r["total_tracks"]
    r["duplication"] = r["total_seeds"] / r["total_tracks"]
    r2 = (
        r.groupby(col)
        .agg(
            {
                "efficiency": ["mean", "std"],
                "duplication": ["mean", "std"],
                "total_tracks": ["mean", "std"],
                "matched_tracks": ["mean", "std"],
            }
        )
        .fillna(0)
    )

    # Plot track counts
    matplotlib.pyplot.figure()
    fig, ax = matplotlib.pyplot.subplots()
    ax.errorbar(
        (bins[1:] + bins[:-1]) / 2,
        r2["total_tracks"]["mean"],
        yerr=r2["total_tracks"]["std"],
        fmt=".",
        xerr=(bins[1:] - bins[:-1]) / 2,
        capsize=2,
        label="Total",
    )
    ax.errorbar(
        (bins[1:] + bins[:-1]) / 2,
        r2["matched_tracks"]["mean"],
        yerr=r2["matched_tracks"]["std"],
        fmt=".",
        xerr=(bins[1:] - bins[:-1]) / 2,
        capsize=2,
        label="Matched",
    )
    if logx:
        ax.set_xscale("log")
    ax.set_ylim(ymin=0)
    ax.legend(loc="upper right")
    ax.set_ylabel("Track count")
    ax.set_title(f"Track count vs {pretty_print(col)}")
    ax.set_xlabel(pretty_print(col))
    pdf.savefig()
    matplotlib.pyplot.close()

    # Plot efficiencies
    matplotlib.pyplot.figure()
    fig, ax = matplotlib.pyplot.subplots()
    ax.errorbar(
        (bins[1:] + bins[:-1]) / 2,
        r2["efficiency"]["mean"],
        yerr=r2["efficiency"]["std"],
        fmt=".",
        xerr=(bins[1:] - bins[:-1]) / 2,
        capsize=2,
    )
    if logx:
        ax.set_xscale("log")
    ax.set_ylim(ymin=0, ymax=1)
    ax.set_ylabel("Efficiency")
    ax.set_title(f"Efficiency vs {pretty_print(col)}")
    ax.set_xlabel(pretty_print(col))
    pdf.savefig()
    matplotlib.pyplot.close()

    # Plot duplicate rate
    matplotlib.pyplot.figure()
    fig, ax = matplotlib.pyplot.subplots()
    ax.errorbar(
        (bins[1:] + bins[:-1]) / 2,
        r2["duplication"]["mean"],
        yerr=r2["duplication"]["std"],
        fmt=".",
        xerr=(bins[1:] - bins[:-1]) / 2,
        capsize=2,
    )
    ax.set_ylim(ymin=0)
    if logx:
        ax.set_xscale("log")
    ax.set_ylabel("Duplication rate")
    ax.set_title(f"Duplication rate vs {pretty_print(col)}")
    ax.set_xlabel(pretty_print(col))
    pdf.savefig()
    matplotlib.pyplot.close()


def run(
    seed_file: pathlib.Path,
    track_file: pathlib.Path,
    eta_bins: BinSpec,
    phi_bins: BinSpec,
    pt_bins: BinSpec,
    output: typing.Optional[pathlib.Path] = None,
):
    seeds = pandas.read_csv(seed_file)
    tracks = pandas.read_csv(track_file)

    rtracks = tracks[(tracks["q"] != 0) & (~numpy.isinf(tracks["eta"]))]

    eta_min = eta_bins[0] or rtracks["eta"].min()
    eta_max = eta_bins[1] or rtracks["eta"].max()
    phi_min = phi_bins[0] or rtracks["phi"].min()
    phi_max = phi_bins[1] or rtracks["phi"].max()
    pt_min = pt_bins[0] or rtracks["pt"].min()
    pt_max = pt_bins[1] or rtracks["pt"].max()

    ftracks = rtracks[
        (rtracks["pt"] >= pt_min)
        & (rtracks["pt"] <= pt_max)
        & (rtracks["eta"] >= eta_min)
        & (rtracks["eta"] <= eta_max)
        & (rtracks["phi"] >= phi_min)
        & (rtracks["phi"] <= phi_max)
    ]

    q = ftracks.join(
        seeds.groupby(["event_id", "particle_id"])["seed_id"].count(),
        on=["event_id", "particle_id"],
    ).rename(columns={"seed_id": "seed_count"})
    q["seed_count"].fillna(0, inplace=True)

    with PdfPages(output) as pdf:
        make_plot(pdf, q, "eta", numpy.linspace(eta_min, eta_max, eta_bins[2] or 51))
        make_plot(pdf, q, "phi", numpy.linspace(phi_min, phi_max, phi_bins[2] or 51))
        make_plot(
            pdf, q, "pt", numpy.geomspace(pt_min, pt_max, pt_bins[2] or 51), logx=True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot performance of a seeding algorithm"
    )

    parser.add_argument(
        "--seeds",
        "-S",
        type=pathlib.Path,
        default="nseed_performance_seeds.csv",
        help="CSV file with seeds",
    )
    parser.add_argument(
        "--tracks",
        "-T",
        type=pathlib.Path,
        default="nseed_performance_tracks.csv",
        help="CSV file with tracks",
    )
    parser.add_argument(
        "--eta",
        type=range_type,
        default=(-2.7, 2.7, 51),
        help="eta range",
        metavar="[MIN]:[MAX][:BINS]",
    )
    parser.add_argument(
        "--phi",
        type=range_type,
        default=(-3.1415, 3.1415, 51),
        help="phi range",
        metavar="[MIN]:[MAX][:BINS]",
    )
    parser.add_argument(
        "--pt",
        type=range_type,
        default=(1, 20, 51),
        help="pT range",
        metavar="[MIN]:[MAX][:BINS]",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=pathlib.Path,
        default="efficiency.pdf",
        help="PDF file to write",
    )

    args = parser.parse_args()

    run(args.seeds, args.tracks, args.eta, args.phi, args.pt, args.output)
