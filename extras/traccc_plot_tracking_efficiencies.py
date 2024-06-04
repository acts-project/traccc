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


def plotEfficiency(graph, col=ROOT.kBlack, opts="AP"):
    """Plot one efficiency histogram"""

    # Set the properties of the histogram.
    graph.SetTitle("")
    graph.GetYaxis().SetTitle("Efficiency")
    graph.GetYaxis().SetTitleSize(0.07)
    graph.GetYaxis().SetTitleOffset(0.6)
    graph.GetYaxis().SetRangeUser(0.0, 1.1)
    graph.GetXaxis().SetTitleSize(0.07)
    graph.GetXaxis().SetTitleOffset(0.6)
    graph.SetLineColor(col)
    graph.SetMarkerColor(col)

    # Draw the histogram.
    graph.Draw(opts)

    # Return gracefully.
    return graph


def main():
    """C(++) style main function for the script"""

    # Set ROOT into batch mode.
    ROOT.gROOT.SetBatch(True)

    # Parse the command line arguments.
    parser = argparse.ArgumentParser(description="Plot track finding efficiencies")
    parser.add_argument(
        "-i",
        "--input",
        nargs=2,
        help="Input ROOT files",
        default="track_finding_efficiency.root",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output image file",
        default="track_finding_efficiency.png",
    )
    args = parser.parse_args()

    # Create a canvas to draw on.
    canvas = ROOT.TCanvas("canvas", "canvas", 1000, 1000)
    canvas.Divide(1, 2)

    # Open the seeding efficiency file.
    seeding_efficiency_file = ROOT.TFile.Open(args.input[0], "READ")
    if not seeding_efficiency_file:
        print(f"Failed to open input file '{args.input[0]}'")
        return 1

    # Get the seeding efficiency histograms.
    seeding_efficiency_eta = seeding_efficiency_file.Get("seed_finding_eff_eta")
    seeding_efficiency_phi = seeding_efficiency_file.Get("seed_finding_eff_phi")
    if not seeding_efficiency_eta or not seeding_efficiency_phi:
        print("Failed to retrieve efficiency histograms in %s" % args.input[0])
        return 1

    # Open the tracking efficiency file.
    tracking_efficiency_file = ROOT.TFile.Open(args.input[1], "READ")
    if not tracking_efficiency_file:
        print(f"Failed to open input file '{args.input[1]}'")
        return 1

    # Get the tracking efficiency histograms.
    tracking_efficiency_eta = tracking_efficiency_file.Get("track_finding_eff_eta")
    tracking_efficiency_phi = tracking_efficiency_file.Get("track_finding_eff_phi")
    if not tracking_efficiency_eta or not tracking_efficiency_phi:
        print("Failed to retrieve efficiency histograms in %s" % args.input[1])
        return 1

    # Draw the histograms.
    canvas.cd(1)
    seeding_legend = plotEfficiency(seeding_efficiency_eta.CreateGraph(), ROOT.kBlue)
    tracking_legend = plotEfficiency(
        tracking_efficiency_eta.CreateGraph(), ROOT.kRed, "P"
    )
    canvas.cd(2)
    plotEfficiency(seeding_efficiency_phi.CreateGraph(), ROOT.kBlue)
    plotEfficiency(tracking_efficiency_phi.CreateGraph(), ROOT.kRed, "P")

    # Create a legend.
    legend = ROOT.TLegend(0.6, 0.2, 0.9, 0.4)
    legend.AddEntry(seeding_legend, "Seed finding efficiency", "lpe")
    legend.AddEntry(tracking_legend, "Track finding efficiency", "lpe")
    legend.Draw()

    # Save the canvas to a file.
    canvas.SaveAs(args.output)

    # Return gracefully.
    return 0


# Execute the main function by default.
if __name__ == "__main__":
    import sys

    sys.exit(main())
