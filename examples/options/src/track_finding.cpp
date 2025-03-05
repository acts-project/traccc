/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/options/track_finding.hpp"

#include "traccc/examples/utils/printable.hpp"
#include "traccc/utils/particle.hpp"

// System include(s).
#include <sstream>

namespace traccc::opts {

/// Convenience namespace shorthand
namespace po = boost::program_options;

track_finding::track_finding() : interface("Track Finding Options") {
    m_pdg_number = m_config.ptc_hypothesis.pdg_num();
    m_track_candidates_range[0] = m_config.min_track_candidates_per_track;
    m_track_candidates_range[1] = m_config.max_track_candidates_per_track;

    m_desc.add_options()(
        "max-num-branches-per-seed",
        po::value(&m_config.max_num_branches_per_seed)
            ->default_value(m_config.max_num_branches_per_seed),
        "Max number of branches per seed");
    m_desc.add_options()(
        "max-num-branches-per-surface",
        po::value(&m_config.max_num_branches_per_surface)
            ->default_value(m_config.max_num_branches_per_surface),
        "Max number of branches per surface");
    m_desc.add_options()("track-candidates-range",
                         po::value(&m_track_candidates_range)
                             ->value_name("MIN:MAX")
                             ->default_value(m_track_candidates_range),
                         "Range of track candidates number");
    m_desc.add_options()(
        "min-step-length-for-next-surface",
        po::value(&m_config.min_step_length_for_next_surface)
            ->default_value(m_config.min_step_length_for_next_surface),
        "Minimum step length that track should make to reach the next surface. "
        "This should be set higher than the overstep tolerance not to make it "
        "stay on the same surface");
    m_desc.add_options()(
        "max-step-counts-for-next-surface",
        po::value<unsigned int>(&m_config.max_step_counts_for_next_surface)
            ->default_value(m_config.max_step_counts_for_next_surface),
        "Maximum step counts that track can make to reach the next surface");
    m_desc.add_options()(
        "chi2-max",
        po::value(&m_config.chi2_max)->default_value(m_config.chi2_max),
        "Maximum Chi suqare that measurements can be included in the track");
    m_desc.add_options()(
        "max-num-skipping-per-cand",
        po::value(&m_config.max_num_skipping_per_cand)
            ->default_value(m_config.max_num_skipping_per_cand),
        "Maximum allowed number of skipped steps per candidate");
    m_desc.add_options()("particle-hypothesis",
                         po::value(&m_pdg_number)->default_value(m_pdg_number),
                         "PDG number for the particle hypothesis");
}

track_finding::operator finding_config() const {
    finding_config out = m_config;

    out.min_track_candidates_per_track = m_track_candidates_range[0];
    out.max_track_candidates_per_track = m_track_candidates_range[1];
    out.ptc_hypothesis =
        detail::particle_from_pdg_number<traccc::scalar>(m_pdg_number);

    return out;
}

std::unique_ptr<configuration_printable> track_finding::as_printable() const {
    auto cat = std::make_unique<configuration_category>(m_description);

    cat->add_child(std::make_unique<configuration_kv_pair>(
        "Max branches per seed",
        std::to_string(m_config.max_num_branches_per_seed)));
    cat->add_child(std::make_unique<configuration_kv_pair>(
        "Max branches at surface",
        std::to_string(m_config.max_num_branches_per_surface)));
    std::ostringstream candidate_ss;
    candidate_ss << m_track_candidates_range;
    cat->add_child(std::make_unique<configuration_kv_pair>(
        "Track candidate range", candidate_ss.str()));
    cat->add_child(std::make_unique<configuration_kv_pair>(
        "Min step length to next surface",
        std::to_string(m_config.min_step_length_for_next_surface) + " mm"));
    cat->add_child(std::make_unique<configuration_kv_pair>(
        "Max step count to next surface",
        std::to_string(m_config.max_step_counts_for_next_surface)));
    cat->add_child(std::make_unique<configuration_kv_pair>(
        "Max Chi2", std::to_string(m_config.chi2_max)));
    cat->add_child(std::make_unique<configuration_kv_pair>(
        "Max holes per candidate",
        std::to_string(m_config.max_num_skipping_per_cand)));
    cat->add_child(std::make_unique<configuration_kv_pair>(
        "PDG number", std::to_string(m_pdg_number)));

    return cat;
}
}  // namespace traccc::opts
