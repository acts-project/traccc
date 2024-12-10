# Detray library, part of the ACTS project (R&D line)
#
# (c) 2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

from .plot_ray_scan import (
    read_ray_scan_data,
    plot_intersection_points_xy,
    plot_intersection_points_rz,
)
from .plot_track_params import (
    read_track_data,
    compare_track_pos_xy,
    compare_track_pos_rz,
    plot_track_pos_dist,
    plot_track_pos_res,
)

# python includes
import pandas as pd
import os


""" Read the detector scan data from files and prepare data frames """


def read_scan_data(inputdir, det_name, momentum, logging):

    # Input data directory
    data_dir = os.fsencode(inputdir)

    ray_scan_intersections_file = ray_scan_track_param_file = ""
    helix_scan_intersections_file = helix_scan_track_param_file = ""

    # Find the data files by naming convention
    for file in os.listdir(data_dir):
        filename = os.fsdecode(file)

        if filename.find(det_name + "_ray_scan_intersections_" + momentum) != -1:
            ray_scan_intersections_file = inputdir + "/" + filename
        elif filename.find(det_name + "_ray_scan_track_parameters_" + momentum) != -1:
            ray_scan_track_param_file = inputdir + "/" + filename
        elif filename.find(det_name + "_helix_scan_intersections_" + momentum) != -1:
            helix_scan_intersections_file = inputdir + "/" + filename
        elif filename.find(det_name + "_helix_scan_track_parameters_" + momentum) != -1:
            helix_scan_track_param_file = inputdir + "/" + filename

    # Read ray scan data
    ray_scan_df = read_ray_scan_data(
        ray_scan_intersections_file, ray_scan_track_param_file, logging
    )

    # Read helix scan data
    helix_scan_df = read_ray_scan_data(
        helix_scan_intersections_file, helix_scan_track_param_file, logging
    )

    return ray_scan_df, helix_scan_df


""" Read the recorded track positions from files and prepare data frames """


def read_navigation_data(inputdir, det_name, momentum, read_cuda, logging):

    # Input data directory
    data_dir = os.fsencode(inputdir)

    ray_truth_file = ray_data_file = ray_data_cuda_file = ""
    helix_truth_file = helix_data_file = helix_data_cuda_file = ""

    # Find the data files by naming convention
    for file in os.listdir(data_dir):
        filename = os.fsdecode(file)

        if (
            read_cuda
            and filename.find(
                det_name + "_ray_navigation_track_params_cuda_" + momentum
            )
            != -1
        ):
            ray_data_cuda_file = inputdir + "/" + filename
        elif filename.find(det_name + "_ray_navigation_track_params_" + momentum) != -1:
            ray_data_file = inputdir + "/" + filename
        elif filename.find(det_name + "_ray_truth_track_params_" + momentum) != -1:
            ray_truth_file = inputdir + "/" + filename
        elif (
            read_cuda
            and filename.find(
                det_name + "_helix_navigation_track_params_cuda_" + momentum
            )
            != -1
        ):
            helix_data_cuda_file = inputdir + "/" + filename
        elif (
            filename.find(det_name + "_helix_navigation_track_params_" + momentum) != -1
        ):
            helix_data_file = inputdir + "/" + filename
        elif filename.find(det_name + "_helix_truth_track_params_" + momentum) != -1:
            helix_truth_file = inputdir + "/" + filename

    ray_df = read_track_data(ray_data_file, logging)
    ray_truth_df = read_track_data(ray_truth_file, logging)
    helix_df = read_track_data(helix_data_file, logging)
    helix_truth_df = read_track_data(helix_truth_file, logging)

    ray_cuda_df = helix_cuda_df = pd.DataFrame({})
    if read_cuda:
        ray_cuda_df = read_track_data(ray_data_cuda_file, logging)
        helix_cuda_df = read_track_data(helix_data_cuda_file, logging)

    return ray_df, ray_truth_df, ray_cuda_df, helix_df, helix_truth_df, helix_cuda_df


""" Plot the data gathered during the navigaiton validation """


def plot_navigation_data(
    args,
    det_name,
    plot_factory,
    data_type,
    df_truth,
    truth_name,
    df_ref,
    ref_name,
    out_format="png",
):

    # xy
    compare_track_pos_xy(
        args,
        det_name,
        data_type,
        plot_factory,
        "png",
        df_truth,
        truth_name,
        "r",
        df_ref,
        ref_name,
        "darkgrey",
    )
    # rz
    compare_track_pos_rz(
        args,
        det_name,
        data_type,
        plot_factory,
        "png",
        df_truth,
        truth_name,
        "r",
        df_ref,
        ref_name,
        "darkgrey",
    )

    # Absolut distance
    plot_track_pos_dist(
        args,
        det_name,
        data_type,
        plot_factory,
        out_format,
        df_truth,
        truth_name,
        df_ref,
        ref_name,
    )

    # Residuals
    plot_track_pos_res(
        args,
        det_name,
        data_type,
        plot_factory,
        out_format,
        df_truth,
        truth_name,
        df_ref,
        ref_name,
        "x",
    )
    plot_track_pos_res(
        args,
        det_name,
        data_type,
        plot_factory,
        out_format,
        df_truth,
        truth_name,
        df_ref,
        ref_name,
        "y",
    )
    plot_track_pos_res(
        args,
        det_name,
        data_type,
        plot_factory,
        out_format,
        df_truth,
        truth_name,
        df_ref,
        ref_name,
        "z",
    )
