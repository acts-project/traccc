/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/options/track_finding.hpp"

// System include(s).
#include <iostream>

namespace traccc::opts {

/// Convenience namespace shorthand
namespace po = boost::program_options;

track_finding::track_finding() : interface("Track Finding Options") {

    m_desc.add_options()("track-candidates-range",
                         po::value(&track_candidates_range)
                             ->value_name("MIN:MAX")
                             ->default_value(track_candidates_range),
                         "Range of track candidates number");
    m_desc.add_options()(
        "min-step-length-for-next-surface",
        po::value(&min_step_length_for_next_surface)
            ->default_value(min_step_length_for_next_surface),
        "Minimum step length that track should make to reach the next surface. "
        "This should be set higher than the overstep tolerance not to make it "
        "stay on the same surface");
    m_desc.add_options()(
        "max-step-counts-for-next-surface",
        po::value<unsigned int>(&max_step_counts_for_next_surface)
            ->default_value(max_step_counts_for_next_surface),
        "Maximum step counts that track can make to reach the next surface");
    m_desc.add_options()(
        "chi2-max", po::value(&chi2_max)->default_value(chi2_max),
        "Maximum Chi suqare that measurements can be included in the track");
    m_desc.add_options()(
        "nmax-per-seed",
        po::value<unsigned int>(&nmax_per_seed)->default_value(nmax_per_seed),
        "Maximum number of branches which each initial seed can have at a "
        "step.");
    m_desc.add_options()(
        "max-num-skipping-per-cand",
        po::value<unsigned int>(&max_num_skipping_per_cand)
            ->default_value(max_num_skipping_per_cand),
        "Maximum allowed number of skipped steps per candidate");
}

track_finding::operator finding_config<float>() const {
    finding_config<float> out;
    out.min_track_candidates_per_track = track_candidates_range[0];
    out.max_track_candidates_per_track = track_candidates_range[1];
    out.min_step_length_for_next_surface = min_step_length_for_next_surface;
    out.max_step_counts_for_next_surface = max_step_counts_for_next_surface;
    out.chi2_max = chi2_max;
    out.max_num_branches_per_seed = nmax_per_seed;
    out.max_num_skipping_per_cand = max_num_skipping_per_cand;
    return out;
}

track_finding::operator finding_config<double>() const {
    finding_config<double> out;
    out.min_track_candidates_per_track = track_candidates_range[0];
    out.max_track_candidates_per_track = track_candidates_range[1];
    out.min_step_length_for_next_surface = min_step_length_for_next_surface;
    out.max_step_counts_for_next_surface = max_step_counts_for_next_surface;
    out.chi2_max = chi2_max;
    out.max_num_branches_per_seed = nmax_per_seed;
    out.max_num_skipping_per_cand = max_num_skipping_per_cand;
    return out;
}

std::ostream& track_finding::print_impl(std::ostream& out) const {

    out << "  Track candidates range   : " << track_candidates_range << "\n"
        << "  Minimum step length for the next surface: "
        << min_step_length_for_next_surface << " [mm] \n"
        << "  Maximum step counts for the next surface: "
        << max_step_counts_for_next_surface << "\n"
        << "  Maximum Chi2             : " << chi2_max << "\n"
        << "  Maximum branches per step: " << nmax_per_seed << "\n"
        << "  Maximum number of skipped steps per candidates: "
        << max_num_skipping_per_cand;
    return out;
}

}  // namespace traccc::opts
