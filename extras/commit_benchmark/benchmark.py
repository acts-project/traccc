import argparse
import sys
import csv
import pandas
import git
import pathlib
import shutil
import logging
import tempfile
import subprocess
import os
import time
import operator
import functools
import numpy


log = logging.getLogger("traccc_benchmark")


DETERMINISTIC_ORDER_COMMIT = "7e7f17ccd2e2b0db8971655773b351a365ee1cfc"


class GpuSpec:
    def __init__(self, n_sm, n_threads_per_sm):
        self.n_sm = n_sm
        self.n_threads_per_sm = n_threads_per_sm


def harmonic_sum(vals):
    return 1.0 / sum(1.0 / x for x in vals)


def parse_triple(triple):
    assert triple[0] == "(" and triple[-1] == ")"
    x, y, z = triple[1:-1].split(", ")
    return int(x), int(y), int(z)


def simplify_name(name):
    if name[:5] == "void ":
        name = name[5:]

    val = ""

    while name:
        if name[:2] == "::":
            val = ""
            name = name[2:]
        elif name[0] == "(" or name[0] == "<":
            return val
        else:
            val = val + name[0]
            name = name[1:]

    raise RuntimeError("An error occured in name simpliciation")


def map_name(name):
    if name in [
        "DeviceRadixSortUpsweepKernel",
        "RadixSortScanBinsKernel",
        "DeviceRadixSortDownsweepKernel",
        "DeviceRadixSortSingleTileKernel",
        "DeviceMergeSortBlockSortKernel",
        "DeviceMergeSortMergeKernel",
        "DeviceMergeSortPartitionKernel",
        "_kernel_agent",
    ]:
        return "Thrust::sort"
    else:
        return name


def parse_profile_csv(file: pathlib.Path, gpu_spec: GpuSpec):
    df = pandas.read_csv(file)

    ndf = df[df["Metric Name"] == "Duration"][
        ["ID", "Kernel Name", "Block Size", "Grid Size", "Metric Value", "Metric Unit"]
    ]

    assert (ndf["Metric Unit"] == "ns").all()

    ndf["ThreadsPerBlock"] = df["Block Size"].apply(
        lambda x: functools.reduce(operator.mul, (parse_triple(x)))
    )
    ndf["BlocksPerGrid"] = df["Grid Size"].apply(
        lambda x: functools.reduce(operator.mul, (parse_triple(x)))
    )
    ndf["TotalThreads"] = ndf["ThreadsPerBlock"] * ndf["BlocksPerGrid"]
    ndf = ndf.drop(
        ["Block Size", "Grid Size", "ThreadsPerBlock", "BlocksPerGrid", "Metric Unit"],
        axis=1,
    )
    ndf["Metric Value"] = ndf["Metric Value"].apply(lambda x: int(x.replace(",", "")))
    ndf["Kernel Name"] = ndf["Kernel Name"].apply(simplify_name)
    ndf["Kernel Name"] = ndf["Kernel Name"].apply(map_name)

    curr_evt_id = None
    evt_ids = []
    for x in ndf.iloc:
        if x["Kernel Name"] == "ccl_kernel":
            if curr_evt_id is None:
                curr_evt_id = 0
            else:
                curr_evt_id += 1
        evt_ids.append(curr_evt_id)

    ndf["EventID"] = evt_ids

    thr_occ = df[df["Metric Name"] == "Theoretical Occupancy"]

    ndf = ndf.merge(
        thr_occ[["ID", "Metric Value"]],
        on="ID",
        how="left",
        validate="one_to_one",
        suffixes=("", "R"),
    )

    ndf["Occupancy"] = ndf["Metric ValueR"].apply(lambda x: float(x) / 100.0)

    ndf["k"] = ndf["TotalThreads"] / (
        gpu_spec.n_sm * gpu_spec.n_threads_per_sm * ndf["Occupancy"]
    )

    ndf["Throughput"] = (numpy.ceil(ndf["k"]) / ndf["k"]) / (ndf["Metric Value"] / 1e9)
    ndf["RecThroughput"] = 1.0 / ndf["Throughput"]

    ndf = ndf.drop(["Metric ValueR"], axis=1)

    ndf = ndf.groupby(["Kernel Name", "EventID"], as_index=False).agg(
        {"Throughput": harmonic_sum, "RecThroughput": "sum"},
    )

    ndf = ndf.groupby("Kernel Name", as_index=False).agg(
        ThroughputMean=("Throughput", "mean"),
        ThroughputStd=("Throughput", "std"),
        RecThroughputMean=("RecThroughput", "mean"),
        RecThroughputStd=("RecThroughput", "std"),
    )

    return ndf.fillna(0)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "repo",
        type=pathlib.Path,
        help="the traccc build repository",
    )

    parser.add_argument(
        "db",
        type=pathlib.Path,
        help="the CSV database",
    )

    parser.add_argument(
        "data",
        type=pathlib.Path,
        help="the traccc dataset to use",
    )

    parser.add_argument(
        "-f",
        "--from",
        type=str,
        help="the first commit in range (exclusive)",
        required=True,
    )

    parser.add_argument(
        "-t",
        "--to",
        type=str,
        help="the last commit in range (inclusive)",
        default="HEAD",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        help="enable verbose output",
        action="store_true",
    )

    parser.add_argument(
        "-j",
        "--parallel",
        help="number of threads to use for compilation",
        default=1,
        type=int,
    )

    parser.add_argument(
        "-e",
        "--events",
        help="number of events to process per commit",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--num-sm",
        help="number of SMs in the modelled GPU",
        type=int,
        required=True,
        dest="num_sm",
    )

    parser.add_argument(
        "--num-threads-per-sm",
        help="number of thread slots per SM in the modelled GPU",
        type=int,
        required=True,
        dest="num_threads_per_sm",
    )

    parser.add_argument(
        "--ncu-wrapper",
        help="wrapper to use around the ncu command",
        type=str,
        dest="ncu_wrapper",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if (args.verbose or False) else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    log.info(
        "Using GPU with %d SMs and %d thread slots per SM (%d thread slots total)",
        getattr(args, "num_sm"),
        getattr(args, "num_threads_per_sm"),
        getattr(args, "num_sm") * getattr(args, "num_threads_per_sm"),
    )

    repo = git.Repo(args.repo)

    log.info("Using git repository at %s", repo.git_dir)

    if repo.is_dirty():
        log.fatal("Repository is dirty; please clean it before use!")
        raise RuntimeError("Repository is dirty; please clean it before use!")

    if "TRACCC_TEST_DATA_DIR" not in os.environ:
        e = 'Environment variable "TRACCC_TEST_DATA_DIR" is not set; aborting!'
        log.fatal(e)
        raise RuntimeError(e)

    for exec in ["ncu", "g++", "nvcc", "cmake"]:
        if shutil.which(exec) is None:
            e = 'Executable "%s" is not available; aborting' % exec
            log.fatal(e)
            raise RuntimeError(e)

    old_commit_hash = repo.head.object.hexsha
    log.info("Current commit hash is %s", old_commit_hash)

    commit_range = repo.iter_commits(
        "%s...%s" % (getattr(args, "from"), getattr(args, "to"))
    )

    commit_list = list(commit_range)
    commit_str_list = list(str(x) for x in commit_list)

    log.info(
        "Examining a total of %d commits between %s and %s",
        len(commit_list),
        getattr(args, "from"),
        getattr(args, "to"),
    )

    if args.db.is_file():
        log.info('Database file "%s" already exists; creating a backup', args.db)
        shutil.copy(str(args.db), str(args.db) + ".bak")
        results = []
        with open(args.db, "r") as f:
            reader = csv.DictReader(f)
            for i in reader:
                results.append(i)
        log.info("Database contained %d pre-existing results", len(results))
    else:
        log.info('Database file "%s" does not exist; starting from scratch', args.db)
        results = []

    known_commits = set(x["commit"] for x in results)

    log.info("Currently have pre-existing results for %d commits", len(known_commits))

    commits_to_skip = set()

    for x in commit_list:
        log.info("Considering commit %s", x)

        parents = x.parents

        # If this commit has a parent with a single parent, and that
        # grandparent is also a parent of this current commit, we have a
        # single-commit merge which we don't need to measure separately.
        if len(parents) == 2:
            for p1 in parents:
                if len(p1.parents) == 1 and p1.parents[0] in parents:
                    log.info("Commit %s is a triangle commit; adding to skip list", p1)
                    commits_to_skip.add(str(p1))

        # Skip commits which we have already benchmarked
        if str(x) in known_commits:
            log.info("Commit %s is already know; skipping", x)
            continue

        if str(x) in commits_to_skip:
            log.info("Commit %s is marked for skipping", x)
            continue

        try:
            log.info("Running benchmark for commit %s", x)

            log.info("Running checkout step")

            start_time = time.time()
            repo.git.checkout(x)
            end_time = time.time()

            log.info("Completed checkout step in %.1f seconds", end_time - start_time)

            with tempfile.TemporaryDirectory() as tmpdirname:
                tmppath = pathlib.Path(tmpdirname)

                log.info('Created temporary directory "%s"', str(tmppath))

                build_dir = tmppath / "build"

                log.info('Building traccc into build directory "%s"', build_dir)

                log.info("Running configuration step")

                start_time = time.time()
                subprocess.run(
                    [
                        "cmake",
                        "-S",
                        args.repo,
                        "-B",
                        build_dir,
                        "-DTRACCC_BUILD_CUDA=ON",
                        "-DCMAKE_BUILD_TYPE=Release",
                        "-DTRACCC_USE_ROOT=OFF",
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
                end_time = time.time()

                log.info(
                    "Completed configuration step in %.1f seconds",
                    end_time - start_time,
                )

                log.info("Running build step with %d thread(s)", args.parallel)

                start_time = time.time()
                subprocess.run(
                    [
                        "cmake",
                        "--build",
                        build_dir,
                        "--",
                        "-j",
                        str(args.parallel),
                        "traccc_throughput_st_cuda",
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
                end_time = time.time()

                log.info("Completed build step in %.1f seconds", end_time - start_time)

                log.info("Running benchmark step")

                start_time = time.time()

                profile_args = [
                    "ncu",
                    "--import-source",
                    "no",
                    "--set",
                    "basic",
                    "-f",
                    "-o",
                    build_dir / "profile",
                    build_dir / "bin" / "traccc_throughput_st_cuda",
                    "--input-directory=%s" % args.data,
                    "--digitization-file=geometries/odd/odd-digi-geometric-config.json",
                    "--detector-file=geometries/odd/odd-detray_geometry_detray.json",
                    "--grid-file=geometries/odd/odd-detray_surface_grids_detray.json",
                    "--use-detray-detector",
                    "--input-events=%d" % min(100, args.events),
                    "--cold-run-events=0",
                    "--processed-events=%d" % args.events,
                    "--use-acts-geom-source",
                ]

                if hasattr(args, "ncu_wrapper") and args.ncu_wrapper is not None:
                    profile_args = getattr(args, "ncu_wrapper").split() + profile_args

                for p in x.iter_parents():
                    if (
                        str(x) == DETERMINISTIC_ORDER_COMMIT
                        or str(p) == DETERMINISTIC_ORDER_COMMIT
                    ):
                        log.info(
                            "Commit is a child of (or is) %s; enabling deterministic processing",
                            DETERMINISTIC_ORDER_COMMIT[:8],
                        )
                        profile_args.append("--deterministic")
                        break
                else:
                    log.info(
                        "Commit is not a child of %s; event order is random",
                        DETERMINISTIC_ORDER_COMMIT[:8],
                    )

                subprocess.run(
                    profile_args,
                    stdout=subprocess.DEVNULL,
                )
                end_time = time.time()

                log.info(
                    "Completed benchmark step in %.1f seconds", end_time - start_time
                )

                log.info("Running CSV conversion step")

                profile_file = build_dir / "profile.csv"

                start_time = time.time()
                with open(profile_file, "w") as f:
                    subprocess.run(
                        [
                            "ncu",
                            "-i",
                            build_dir / "profile.ncu-rep",
                            "--csv",
                            "--print-units",
                            "base",
                        ],
                        stdout=f,
                    )
                end_time = time.time()

                log.info(
                    "Completed CSV conversion step in %.1f seconds",
                    end_time - start_time,
                )

                log.info("Running data processing step")

                start_time = time.time()
                result_df = parse_profile_csv(
                    profile_file,
                    GpuSpec(
                        getattr(args, "num_sm"), getattr(args, "num_threads_per_sm")
                    ),
                )
                end_time = time.time()

                log.info(
                    "Completed data processing step in %.1f seconds",
                    end_time - start_time,
                )

            for y in result_df.iloc:
                results.append(
                    {
                        "commit": str(x),
                        "kernel": y["Kernel Name"],
                        "throughput": y["ThroughputMean"],
                        "throughput_dev": y["ThroughputStd"],
                        "rec_throughput": y["RecThroughputMean"],
                        "rec_throughput_dev": y["RecThroughputStd"],
                    }
                )

        except Exception as e:
            log.exception(e)

    log.info("Gathered a total of %d results (incl. pre-existing)", len(results))
    output_results = sorted(
        [x for x in results if x["commit"] in commit_str_list],
        key=lambda x: -commit_str_list.index(x["commit"]),
    )
    log.info("Keeping a total of %d results after pruning", len(output_results))

    log.info("Writing data to %s", args.db)
    with open(args.db, "w") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "commit",
                "kernel",
                "throughput",
                "throughput_dev",
                "rec_throughput",
                "rec_throughput_dev",
            ],
        )
        writer.writeheader()

        for i in output_results:
            writer.writerow(i)

    log.info("Checking out repository to previous commit %s", old_commit_hash)
    repo.git.checkout(old_commit_hash)
    log.info("Processing complete; goodbye!")


if __name__ == "__main__":
    main()
