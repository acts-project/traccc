# SPDX-PackageName = "traccc, a part of the ACTS project"
# SPDX-FileCopyrightText: CERN
# SPDX-License-Identifier: MPL-2.0

import argparse
import sys
import csv
import git
import pathlib
import shutil
import logging
import tempfile
import subprocess
import os
import time

from traccc_bench_tools import parse_profile, types


log = logging.getLogger("traccc_benchmark")


DETERMINISTIC_ORDER_COMMIT = "7e7f17ccd2e2b0db8971655773b351a365ee1cfc"
BOOLEAN_FLAG_COMMIT = "380fc78ba63a79ed5c8f19d01d57636aa31cf4fd"
SPACK_LIBS_COMMIT = "069cc80b845c16bf36430fdc90130f0306b47f3e"


def is_parent_of(subj, parent_str):
    for p in subj.iter_parents():
        if str(subj) == parent_str or str(p) == parent_str:
            return True
    return False


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

                config_args = [
                    "cmake",
                    "-S",
                    args.repo,
                    "-B",
                    build_dir,
                    "-DTRACCC_BUILD_CUDA=ON",
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DTRACCC_USE_ROOT=OFF",
                ]

                if is_parent_of(x, SPACK_LIBS_COMMIT):
                    log.info(
                        "Commit is a child of (or is) %s; enabling Spack libraries",
                        SPACK_LIBS_COMMIT[:8],
                    )
                    config_args.append("-DTRACCC_USE_SPACK_LIBS=ON")
                else:
                    log.info(
                        "Commit is not a child of %s; disabling Spack libraries",
                        SPACK_LIBS_COMMIT[:8],
                    )
                    config_args.append("-DTRACCC_USE_SYSTEM_ACTS=ON")
                    config_args.append("-DTRACCC_USE_SYSTEM_TBB=ON")

                subprocess.run(
                    config_args,
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
                    "--section LaunchStats",
                    "--section Occupancy",
                    "--metrics gpu__time_duration.sum",
                    "-f",
                    "-o",
                    build_dir / "profile",
                    build_dir / "bin" / "traccc_throughput_st_cuda",
                    "--input-directory=%s" % args.data,
                    "--digitization-file=geometries/odd/odd-digi-geometric-config.json",
                    "--detector-file=geometries/odd/odd-detray_geometry_detray.json",
                    "--grid-file=geometries/odd/odd-detray_surface_grids_detray.json",
                    "--input-events=%d" % min(100, args.events),
                    "--cold-run-events=0",
                    "--processed-events=%d" % args.events,
                ]

                if hasattr(args, "ncu_wrapper") and args.ncu_wrapper is not None:
                    profile_args = getattr(args, "ncu_wrapper").split() + profile_args

                if is_parent_of(x, DETERMINISTIC_ORDER_COMMIT):
                    log.info(
                        "Commit is a child of (or is) %s; enabling deterministic processing",
                        DETERMINISTIC_ORDER_COMMIT[:8],
                    )
                    profile_args.append("--deterministic")
                else:
                    log.info(
                        "Commit is not a child of %s; event order is random",
                        DETERMINISTIC_ORDER_COMMIT[:8],
                    )

                if is_parent_of(x, BOOLEAN_FLAG_COMMIT):
                    log.info(
                        "Commit is a child of (or is) %s; using explicit boolean flags",
                        BOOLEAN_FLAG_COMMIT[:8],
                    )
                    profile_args.append("--use-acts-geom-source=1")
                    profile_args.append("--use-detray-detector=1")
                else:
                    log.info(
                        "Commit is not a child of %s; using implicit boolean flags",
                        BOOLEAN_FLAG_COMMIT[:8],
                    )
                    profile_args.append("--use-acts-geom-source")
                    profile_args.append("--use-detray-detector")

                subprocess.run(
                    profile_args,
                    stdout=subprocess.DEVNULL,
                )
                end_time = time.time()

                log.info(
                    "Completed benchmark step in %.1f seconds", end_time - start_time
                )

                log.info("Running data processing step")

                start_time = time.time()
                result_df = parse_profile.parse_profile_ncu(
                    build_dir / "profile.ncu-rep",
                    types.GpuSpec(
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
        except KeyboardInterrupt as e:
            log.info("Received keyboard interrupt; skipping to post-processing")
            break

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
