#!/usr/bin/env python3

import pandas as pd
import argparse
import pathlib
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from numpy import log2
import pickle


def processData(inputs):
    finals = []
    peaks = pd.DataFrame()
    for input_csv in inputs:
        # Read raw input data
        raw = pd.read_csv(input_csv)
        print("Read ", len(raw.index), " datapoints from ", input_csv)

        # Get int value from input directory name
        raw["mu"] = raw["directory"].str.extract("(\d+)").astype(int)

        # Calculate throughput
        raw["throughput (events/s)"] = (
            raw["processed_events"] * 1e9 / raw["processing_time"]
        )

        # Calculate mean throughput and respective standard deviation for each mu/thread pair
        clean = (
            raw.groupby(["mu", "threads"])
            .agg({"throughput (events/s)": [("stddev", "std"), ("mean", "mean")]})
            .fillna(0.0)
        )

        clean["input_csv"] = input_csv
        finals.append(clean)

        # Add maximum values to global comparison table
        peak = clean.loc[
            clean["throughput (events/s)"]["mean"].groupby("mu").idxmax()
        ].reset_index("threads")

        peaks[input_csv] = peak["throughput (events/s)"]["mean"]
        peaks["stddev_{}".format(input_csv)] = peak["throughput (events/s)"]["stddev"]

    df = pd.concat(finals)
    return df, peaks


def clearScreen():
    # For Windows
    if os.name == "nt":
        _ = os.system("cls")
    # For macOS and Linux
    else:
        _ = os.system("clear")


def myLog2Format(x, pos):
    decimalplaces = int(max(-log2(x), 0))  # =0 for numbers >=1
    formatstring = "{{:.{:1d}f}}".format(decimalplaces)
    return formatstring.format(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="throughput data plotting")

    parser.add_argument(
        "inputs",
        nargs="+",
        type=pathlib.Path,
        help="input CSV files",
    )
    parser.add_argument(
        "output",
        type=pathlib.Path,
        help="output directory",
    )

    args = parser.parse_args()
    inputs = args.inputs

    df, peaks = processData(inputs)

    displayNames = dict()
    try:
        with open(".plot_data_dictionary.pkl", "rb") as f:
            displayNames = pickle.load(f)
    except:
        pass

    for inpt in inputs:
        if not (inpt in displayNames):
            displayNames[inpt] = str(inpt.stem)

    while True:
        clearScreen()
        print(
            'To edit the displayed names of data by writing "<index> <displayed name> [ENTER]". Or just hit [ENTER] when finished.\n\n'
            + "index | file | displayed name | index"
        )
        for i in range(len(inputs)):
            print("{} {} {} {}".format(i, inputs[i], displayNames[inputs[i]], i))
        print("\n\n")
        inpt = input().split(" ")
        if len(inpt) <= 1:
            break
        displayNames[list(inputs)[int(inpt[0])]] = " ".join(inpt[1:])

    with open(".plot_data_dictionary.pkl", "wb") as f:
        pickle.dump(displayNames, f)

    markers = [".", "x", "v", "s", "*", "^", "o", "d", "p", "H"]
    linestyles = ["-", "--", "-.", ":"]

    print("Printing per-hardware performance")
    for inpt in inputs:
        for mu in df[df["input_csv"] == inpt].index.get_level_values("mu").unique():
            this_df = df[df["input_csv"] == inpt].loc[mu,]
            plt.errorbar(
                x=this_df.index,
                y=this_df[[("throughput (events/s)", "mean")]].values[:, 0],
                yerr=this_df[[("throughput (events/s)", "stddev")]].values[:, 0],
                label="mu{}".format(str(mu)),
            )
        plt.legend(loc="lower right", fontsize="6")
        plt.title(displayNames[inpt])
        plt.xlabel("Threads")
        plt.ylabel("Throughput (events/s)")
        plt.savefig(args.output.joinpath("{}.pdf".format(displayNames[inpt])))
        plt.close()

    print("Printing across-hardware performance")
    for mu in df.index.get_level_values("mu").unique():
        fig, ax = plt.subplots()
        for i in range(len(inputs)):
            this_df = df[df["input_csv"] == inputs[i]].loc[mu,]
            if len(this_df.index) > 0:
                plt.errorbar(
                    x=this_df.index,
                    y=this_df[[("throughput (events/s)", "mean")]].values[:, 0],
                    yerr=this_df[[("throughput (events/s)", "stddev")]].values[:, 0],
                    label=displayNames[inputs[i]],
                    marker=markers[i % len(markers)],
                    markersize=5,
                    linestyle=linestyles[i % len(linestyles)],
                )
        plt.legend(loc="lower right", fontsize="6")
        plt.title("mu{}".format(mu))
        plt.xlabel("Threads")
        plt.ylabel("Throughput (events/s)")
        plt.savefig(args.output.joinpath("allHardware_mu{}.pdf".format(str(mu))))
        plt.xscale("log", base=2)
        ax.xaxis.set_major_formatter(FuncFormatter(myLog2Format))
        plt.savefig(args.output.joinpath("allHardware_mu{}_xlog2.pdf".format(str(mu))))
        plt.close()

    print("Printing across-hardware peak performance")
    for i in range(len(inputs)):
        plt.errorbar(
            x=peaks.index,
            y=peaks[[inputs[i]]].values[:, 0],
            yerr=peaks[["stddev_{}".format(inputs[i])]].values[:, 0],
            label=displayNames[inputs[i]],
            marker=markers[i % len(markers)],
            markersize=5,
            linestyle=linestyles[i % len(linestyles)],
        )
    plt.legend(loc="upper right", fontsize="6")
    plt.title("Maximum throughput across hardware")
    plt.xlabel("ttbar pile-up")
    plt.ylabel("Throughput (events/s)")
    plt.savefig(args.output.joinpath("peaks.pdf"))
    plt.yscale("log")
    plt.savefig(args.output.joinpath("peaks_ylog.pdf"))
    plt.close()
