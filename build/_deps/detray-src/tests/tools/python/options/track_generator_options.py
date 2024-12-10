# Detray library, part of the ACTS project (R&D line)
#
# (c) 2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

import argparse

# -------------------------------------------------------------------------------
# Options parsing
# -------------------------------------------------------------------------------

""" Parent parser that contains propagation options """


def random_track_generator_options():

    parser = argparse.ArgumentParser(add_help=False)

    # Navigation options
    parser.add_argument(
        "--n_tracks", "-n", help=("Number of test tracks"), default=100, type=int
    )
    parser.add_argument(
        "--transverse-momentum",
        "-p_T",
        help=("Transverse momentum of the test particle [GeV]"),
        default=10,
        type=float,
    )
    parser.add_argument(
        "--eta_range",
        "-eta",
        nargs=2,
        help=("Eta range of generated tracks"),
        default=[-4, 4],
        type=float,
    )
    parser.add_argument(
        "--randomize_charge",
        "-rand_chrg",
        help=("Randomly flip charge sign per track"),
        action="store_true",
        default=True,
    )

    return parser


""" Parent parser that contains propagation options """


def uniform_track_generator_options():

    parser = argparse.ArgumentParser(add_help=False)

    # Navigation options
    parser.add_argument(
        "--phi_steps", "-n_phi", help=("Number steps in phi"), default=50, type=int
    )
    parser.add_argument(
        "--eta_steps", "-n_eta", help=("Number steps in eta"), default=50, type=int
    )
    parser.add_argument(
        "--transverse-momentum",
        "-p_T",
        help=("Transverse momentum of the test particle [GeV]"),
        default=10,
        type=float,
    )
    parser.add_argument(
        "--eta_range",
        "-eta",
        nargs=2,
        help=("Eta range of generated tracks"),
        default=[-4, 4],
        type=float,
    )

    return parser


# -------------------------------------------------------------------------------
