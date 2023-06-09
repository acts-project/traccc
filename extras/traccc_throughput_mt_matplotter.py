#!/usr/bin/env python3

import pandas as pd
import argparse
import pathlib
import os
import matplotlib.pyplot as plt
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
            'To edit the displayed names of data by writing "<index> <displayed name> [ENTER]". Or just hit [ENTER] when finished.\n'
        )
        for i in range(len(inputs)):
            print("{} {}".format(i, displayNames[inputs[i]]))
        print("\n\n")
        inpt = input().split(" ")
        if len(inpt) <= 1:
            break
        displayNames[list(inputs)[int(inpt[0])]] = " ".join(inpt[1:])

    with open(".plot_data_dictionary.pkl", "wb") as f:
        pickle.dump(displayNames, f)

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
        plt.savefig(args.output.joinpath("{}.pdf".format(displayNames[inpt])))
        plt.close()

    print("Printing across-hardware performance")
    for mu in df.index.get_level_values("mu").unique():
        for inpt in inputs:
            this_df = df[df["input_csv"] == inpt].loc[mu,]
            if len(this_df.index) > 0:
                plt.errorbar(
                    x=this_df.index,
                    y=this_df[[("throughput (events/s)", "mean")]].values[:, 0],
                    yerr=this_df[[("throughput (events/s)", "stddev")]].values[:, 0],
                    label=displayNames[inpt],
                )
        plt.legend(loc="lower right", fontsize="6")
        plt.savefig(args.output.joinpath("allHardware_mu{}.pdf".format(str(mu))))
        plt.xscale("log", base=2)
        plt.savefig(args.output.joinpath("allHardware_mu{}_xlog2.pdf".format(str(mu))))
        plt.close()

    print("Printing across-hardware peak performance")
    for inpt in inputs:
        if inpt in peaks:
            plt.errorbar(
                x=peaks.index,
                y=peaks[[inpt]].values[:, 0],
                yerr=peaks[["stddev_{}".format(inpt)]].values[:, 0],
                label=displayNames[inpt],
            )
    plt.legend(loc="upper right", fontsize="6")
    plt.savefig(args.output.joinpath("peaks.pdf"))
    plt.yscale("log")
    plt.savefig(args.output.joinpath("peaks_ylog.pdf"))
    plt.close()

