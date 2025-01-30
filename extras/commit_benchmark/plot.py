import matplotlib.pyplot
import pandas
import argparse
import logging
import pathlib

log = logging.getLogger("traccc_plot")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input",
        type=pathlib.Path,
        help="input CSV database",
    )

    parser.add_argument(
        "output",
        type=pathlib.Path,
        help="output plot file",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        help="value in seconds under which to hide kernels",
        default=0.001,
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    log.info('Loading CSV database "%s"', args.input)
    df = pandas.read_csv(args.input)

    valid_kernels = set()

    commits = []

    log.info("Filtering kernels with a threshold of %f seconds", args.threshold)

    for x in df.iloc:
        if x["rec_throughput"] >= args.threshold:
            valid_kernels.add(x["kernel"])
        if x["commit"] not in commits:
            commits.append(x["commit"])

    log.info("Plotting %d kernels over %d commits", len(valid_kernels), len(commits))

    sorted_valid_kernels = sorted(list(valid_kernels))

    f, a = matplotlib.pyplot.subplots(1, 1, figsize=(8, 8))

    xrange = list(range(len(commits)))

    for x in sorted_valid_kernels:
        x_data = []
        y_data = []
        y_error = []

        for y in df.iloc:
            if y["kernel"] == x:
                x_data.append(commits.index(y["commit"]))
                y_data.append(y["rec_throughput"] * 1000)
                y_error.append(y["rec_throughput_dev"] * 1000)

        a.errorbar(
            x_data, y_data, yerr=y_error, label=x, marker="x", capsize=2, elinewidth=1
        )

    a.set_xticks(
        xrange, [x[:8] for x in commits], rotation="vertical", family="monospace"
    )
    a.set_xlabel("Commit hash")
    a.set_ylabel("Reciprocal throughput [ms]")
    a.legend(prop={"family": "monospace"})
    a.grid(color="lightgray", linewidth=0.5)
    f.tight_layout()

    log.info('Saving plot to "%s"', args.output)

    f.savefig(args.output)


if __name__ == "__main__":
    main()
