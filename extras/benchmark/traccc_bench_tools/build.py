# SPDX-PackageName = "traccc, a part of the ACTS project"
# SPDX-FileCopyrightText: CERN
# SPDX-License-Identifier: MPL-2.0

import pathlib
import logging
import subprocess
import typing
import os

from . import git


log = logging.getLogger("traccc_benchmark")


SPACK_LIBS_COMMIT = "069cc80b845c16bf36430fdc90130f0306b47f3e"


def configure(
    source_dir: pathlib.Path, build_dir: pathlib.Path, commit, cc: str = None
):
    config_args = [
        "cmake",
        "-S",
        source_dir,
        "-B",
        build_dir,
        "-DTRACCC_BUILD_CUDA=ON",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DTRACCC_USE_ROOT=OFF",
    ]

    if cc is not None:
        config_args.append("-DCMAKE_CUDA_ARCHITECTURES=%s" % cc)

    if git.is_parent_of(commit, SPACK_LIBS_COMMIT):
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


def build(build_dir: pathlib.Path, parallel: int = 1):
    subprocess.run(
        [
            "cmake",
            "--build",
            build_dir,
            "--",
            "-j",
            str(parallel),
            "traccc_throughput_st_cuda",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
