# Detray library, part of the ACTS project (R&D line)
#
# (c) 2023-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# detray imports
from impl import read_scan_data, read_navigation_data, plot_navigation_data
from impl import plot_track_params
from impl import plot_detector_scan_data, plot_track_pos_dist, plot_track_pos_res
from options import (
    common_options,
    detector_io_options,
    random_track_generator_options,
    propagation_options,
    plotting_options,
)
from options import (
    parse_common_options,
    parse_detector_io_options,
    parse_plotting_options,
)
from plotting import pyplot_factory as plt_factory

# python imports
import argparse
import os
import subprocess
import sys
import json


def __main__():

    # ----------------------------------------------------------------arg parsing

    descr = "Detray Navigation Validation"

    # Define options
    parent_parsers = [
        common_options(descr),
        detector_io_options(),
        random_track_generator_options(),
        propagation_options(),
        plotting_options(),
    ]

    parser = argparse.ArgumentParser(description=descr, parents=parent_parsers)

    parser.add_argument(
        "--bindir",
        "-bin",
        help=("Directoy containing the validation executables"),
        default="./bin",
        type=str,
    )
    parser.add_argument(
        "--datadir",
        "-data",
        help=("Directoy containing the data files"),
        default="./validation_data",
        type=str,
    )
    parser.add_argument(
        "--cuda",
        help=("Run the CUDA navigation validation."),
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--sycl",
        help=("Run the SYCL navigation validation."),
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--z_range",
        "-zrng",
        nargs=2,
        help=("z range for the xy-view [mm]."),
        default=[-50, 50],
        type=float,
    )
    parser.add_argument(
        "--hide_portals",
        help=("Hide portal surfaces in plots."),
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--hide_passives",
        help=("Hide passive surfaces in plots."),
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--outlier",
        "-out",
        help=("Threshold for outliers in residual plots [mm]."),
        default=1,
        type=float,
    )

    # Parse options
    args = parser.parse_args()

    logging = parse_common_options(args, descr)
    parse_detector_io_options(args, logging)
    _, out_dir, out_format = parse_plotting_options(args, logging)

    # IO path for data files
    datadir = args.datadir.strip("/")

    # Check bin path
    bindir = args.bindir.strip("/")
    cpu_validation = bindir + "/detray_detector_validation"
    cuda_validation = bindir + "/detray_detector_validation_cuda"

    if not os.path.isdir(bindir) or not os.path.isfile(cpu_validation):
        logging.error(f"Navigation validation binaries were not found! ({args.bindir})")
        sys.exit(1)

    # ------------------------------------------------------------------------run

    # Pass on the options for the validation tools
    args_list = [
        "--data_dir",
        datadir,
        "--geometry_file",
        args.geometry_file,
        "--n_tracks",
        str(args.n_tracks),
        "--randomize_charge",
        str(args.randomize_charge),
        "--p_T",
        str(args.transverse_momentum),
        "--eta_range",
        str(args.eta_range[0]),
        str(args.eta_range[1]),
        "--min_mask_tolerance",
        str(args.min_mask_tol),
        "--max_mask_tolerance",
        str(args.max_mask_tol),
        "--mask_tolerance_scalor",
        str(args.mask_tol_scalor),
        "--overstep_tolerance",
        str(args.overstep_tol),
        "--path_tolerance",
        str(args.path_tol),
        "--rk-tolerance",
        str(args.rk_error_tol),
        "--path_limit",
        str(args.path_limit),
        "--search_window",
        str(args.search_window[0]),
        str(args.search_window[1]),
    ]

    if args.grid_file:
        args_list = args_list + ["--grid_file", args.grid_file]

    if args.material_file:
        args_list = args_list + ["--material_file", args.material_file]

    # Run the host validation and produce the truth data
    logging.debug("Running CPU validation")
    subprocess.run([cpu_validation, "--write_scan_data"] + args_list)

    # Run the device validation (if it has been built)
    if args.cuda and os.path.isfile(cuda_validation):
        logging.debug("Running CUDA validation")
        subprocess.run([cuda_validation] + args_list)

    elif args.cuda:
        logging.error("Could not find CUDA navigation validation executable")

    if args.sycl:
        logging.error("SYCL validation is not implemented")

    # ------------------------------------------------------------------------plot

    logging.info("Generating data plots...\n")

    geo_file = open(args.geometry_file)
    json_geo = json.loads(geo_file.read())

    det_name = json_geo["header"]["common"]["detector"]
    logging.debug("Detector: " + det_name)

    # Check the data path (should have been created when running the validation)
    if not os.path.isdir(datadir):
        logging.error(f"Data directory was not found! ({args.datadir})")
        sys.exit(1)

    plot_factory = plt_factory(out_dir, logging)

    # Read the truth data
    ray_scan_df, helix_scan_df = read_scan_data(
        datadir, det_name, str(args.transverse_momentum), logging
    )

    plot_detector_scan_data(args, det_name, plot_factory, "ray", ray_scan_df, "png")
    plot_detector_scan_data(args, det_name, plot_factory, "helix", helix_scan_df, "png")

    # Plot distributions of track parameter values
    # Only take initial track parameters from generator
    ray_intial_trk_df = ray_scan_df.drop_duplicates(subset=["track_id"])
    helix_intial_trk_df = helix_scan_df.drop_duplicates(subset=["track_id"])
    plot_track_params(
        args, det_name, "helix", plot_factory, out_format, helix_intial_trk_df
    )
    plot_track_params(
        args, det_name, "ray", plot_factory, out_format, ray_intial_trk_df
    )

    # Read the recorded data
    (
        ray_nav_df,
        ray_truth_df,
        ray_nav_cuda_df,
        helix_nav_df,
        helix_truth_df,
        helix_nav_cuda_df,
    ) = read_navigation_data(
        datadir, det_name, str(args.transverse_momentum), args.cuda, logging
    )

    # Plot
    label_cpu = "navigation (CPU)"
    label_cuda = "navigation (CUDA)"

    plot_navigation_data(
        args,
        det_name,
        plot_factory,
        "ray",
        ray_truth_df,
        "truth",
        ray_nav_df,
        label_cpu,
        out_format,
    )

    plot_navigation_data(
        args,
        det_name,
        plot_factory,
        "helix",
        helix_truth_df,
        "truth",
        helix_nav_df,
        label_cpu,
        out_format,
    )

    if args.cuda:
        # Truth vs. Device
        plot_navigation_data(
            args,
            det_name,
            plot_factory,
            "ray",
            ray_truth_df,
            "truth",
            ray_nav_cuda_df,
            label_cuda,
            out_format,
        )

        plot_navigation_data(
            args,
            det_name,
            plot_factory,
            "helix",
            helix_truth_df,
            "truth",
            helix_nav_cuda_df,
            label_cuda,
            out_format,
        )

        # Host vs. Device
        plot_navigation_data(
            args,
            det_name,
            plot_factory,
            "ray",
            ray_nav_df,
            label_cpu,
            ray_nav_cuda_df,
            label_cuda,
            out_format,
        )

        plot_navigation_data(
            args,
            det_name,
            plot_factory,
            "helix",
            helix_nav_df,
            label_cpu,
            helix_nav_cuda_df,
            label_cuda,
            out_format,
        )


# -------------------------------------------------------------------------------

if __name__ == "__main__":
    __main__()

# -------------------------------------------------------------------------------
