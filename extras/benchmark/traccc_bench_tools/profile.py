# SPDX-PackageName = "traccc, a part of the ACTS project"
# SPDX-FileCopyrightText: CERN
# SPDX-License-Identifier: MPL-2.0

import subprocess
import pathlib
import logging
import os

from . import git

log = logging.getLogger("traccc_benchmark")


DETERMINISTIC_ORDER_COMMIT = "7e7f17ccd2e2b0db8971655773b351a365ee1cfc"
BOOLEAN_FLAG_COMMIT = "380fc78ba63a79ed5c8f19d01d57636aa31cf4fd"


def run_profile(
    build_dir: pathlib.Path, data_dir: str, commit, events=1, ncu_wrapper=None
):
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
        "--input-directory=%s" % data_dir,
        "--digitization-file=geometries/odd/odd-digi-geometric-config.json",
        "--detector-file=geometries/odd/odd-detray_geometry_detray.json",
        "--grid-file=geometries/odd/odd-detray_surface_grids_detray.json",
        "--input-events=%d" % events,
        "--cold-run-events=0",
        "--processed-events=%d" % events,
    ]

    if ncu_wrapper is not None:
        profile_args = ncu_wrapper.split() + profile_args

    if git.is_parent_of(commit, DETERMINISTIC_ORDER_COMMIT):
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

    if git.is_parent_of(commit, BOOLEAN_FLAG_COMMIT):
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
        check=True,
    )
