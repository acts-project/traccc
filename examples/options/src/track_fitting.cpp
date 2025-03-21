/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/options/track_fitting.hpp"

#include "traccc/examples/utils/printable.hpp"
#include "traccc/utils/particle.hpp"

// System include(s).
#include <format>

namespace traccc::opts {

/// Convenience namespace shorthand
namespace po = boost::program_options;

track_fitting::track_fitting() : interface("Track Fitting Options") {

    m_desc.add_options()(
        "fit-num-iterations",
        po::value(&m_config.n_iterations)->default_value(m_config.n_iterations),
        "Number of iterations for the track fit");
    m_desc.add_options()(
        "fit-particle-hypothesis",
        po::value(&m_pdg)->value_name("PDG")->default_value(m_pdg),
        "Particle hypothesis for the track fit");
    m_desc.add_options()("fit-use-backward-filter",
                         po::value(&m_config.use_backward_filter)
                             ->default_value(m_config.use_backward_filter),
                         "Use backward filter for smoothing");
    m_desc.add_options()(
        "fit-covariance-inflation-factor",
        po::value(&m_config.covariance_inflation_factor)
            ->default_value(m_config.covariance_inflation_factor),
        "Covariance inflation factor for the track fit");
    m_desc.add_options()(
        "barcode-sequence-size-factor",
        po::value(&m_config.barcode_sequence_size_factor)
            ->default_value(m_config.barcode_sequence_size_factor),
        "Size factor for the barcode sequence used in the backward filter");
    m_desc.add_options()(
        "min-barcode-sequence-capacity",
        po::value(&m_config.min_barcode_sequence_capacity)
            ->default_value(m_config.min_barcode_sequence_capacity),
        "Minimum capacity of barcode sequence");
    m_desc.add_options()(
        "backward-filter-mask-tolerance",
        po::value(&m_config.backward_filter_mask_tolerance)
            ->default_value(m_config.backward_filter_mask_tolerance),
        "Mask tolerance for the backward filter");
}

track_fitting::operator fitting_config() const {

    fitting_config out = m_config;

    out.ptc_hypothesis =
        detail::particle_from_pdg_number<traccc::scalar>(m_pdg);

    return out;
}

std::unique_ptr<configuration_printable> track_fitting::as_printable() const {
    auto cat = std::make_unique<configuration_category>(m_description);

    cat->add_child(std::make_unique<configuration_kv_pair>(
        "Number of iterations", std::to_string(m_config.n_iterations)));
    cat->add_child(std::make_unique<configuration_kv_pair>(
        "Particle hypothesis PDG", std::to_string(m_pdg)));
    cat->add_child(std::make_unique<configuration_kv_pair>(
        "Use backward filter",
        std::format("{}", m_config.use_backward_filter)));
    cat->add_child(std::make_unique<configuration_kv_pair>(
        "Covariance inflation factor",
        std::to_string(m_config.covariance_inflation_factor)));
    cat->add_child(std::make_unique<configuration_kv_pair>(
        "Barcode sequence size factor",
        std::to_string(m_config.barcode_sequence_size_factor)));
    cat->add_child(std::make_unique<configuration_kv_pair>(
        "Minimum capacity of barcode sequence",
        std::to_string(m_config.min_barcode_sequence_capacity)));
    cat->add_child(std::make_unique<configuration_kv_pair>(
        "Mask tolerance for the backward filter",
        std::to_string(m_config.backward_filter_mask_tolerance)));

    return cat;
}
}  // namespace traccc::opts
