#!/usr/bin/env python3
#
# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0
#

# Import the required modules.
import argparse
import ROOT


def plotEfficiency(graph):
    """Plot one efficiency histogram"""

    # Set the properties of the histogram.
    graph.SetTitle("")
    graph.GetYaxis().SetTitle("Efficiency")
    graph.GetYaxis().SetTitleSize(0.07)
    graph.GetYaxis().SetTitleOffset(0.6)
    graph.GetYaxis().SetRangeUser(0.0, 1.1)
    graph.GetXaxis().SetTitleSize(0.07)
    graph.GetXaxis().SetTitleOffset(0.6)

    # Draw the histogram.
    graph.Draw("AP")

    # Return gracefully.
    return


def main():
    """C(++) style main function for the script"""

    # Set ROOT into batch mode.
    ROOT.gROOT.SetBatch(True)

    # Parse the command line arguments.
    parser = argparse.ArgumentParser(description="Plot track finding efficiencies")
    parser.add_argument(
        "-i",
        "--input",
        help="Input ROOT file",
        default="track_finding_efficiency.root",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output image file",
        default="track_finding_efficiency.png",
    )
    args = parser.parse_args()

    # Open the input file.
    input_file = ROOT.TFile.Open(args.input, "READ")
    if not input_file:
        print(f"Failed to open input file '{args.input}'")
        return 1

    # Create a canvas to draw on.
    canvas = ROOT.TCanvas("canvas", "canvas", 1000, 1000)
    canvas.Divide(1, 2)

    # Get the efficiency histograms from the input file.
    eta_plot = input_file.Get("track_finding_eff_eta")
    phi_plot = input_file.Get("track_finding_eff_phi")
    if not eta_plot or not phi_plot:
        print("Failed to retrieve efficiency histograms")
        return 1

    # Draw the histograms.
    canvas.cd(1)
    plotEfficiency(eta_plot.CreateGraph())
    canvas.cd(2)
    plotEfficiency(phi_plot.CreateGraph())

    # Save the canvas to a file.
    canvas.SaveAs(args.output)

    # Return gracefully.
    return 0


# Execute the main function by default.
if __name__ == "__main__":
    import sys

    sys.exit(main())
