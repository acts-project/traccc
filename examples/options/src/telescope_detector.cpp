/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/options/telescope_detector.hpp"

#include "traccc/examples/utils/printable.hpp"

// Detray include(s).
#include "detray/definitions/units.hpp"

// System include(s).
#include <iostream>

namespace traccc::opts {

/// Convenience namespace shorthand
namespace po = boost::program_options;

telescope_detector::telescope_detector()
    : interface("Telescope Detector Options") {

    m_desc.add_options()("empty-material", po::bool_switch(&empty_material),
                         "Build detector without materials");
    m_desc.add_options()("n-planes",
                         po::value(&n_planes)->default_value(n_planes),
                         "Number of planes");
    m_desc.add_options()("thickness-mm",
                         po::value(&thickness)->default_value(thickness),
                         "Slab thickness in [mm]");
    m_desc.add_options()("spacing", po::value(&spacing)->default_value(spacing),
                         "Space between planes in [mm]");
    m_desc.add_options()("smearing-um",
                         po::value(&smearing)->default_value(smearing),
                         "Measurement smearing in [um]");
    m_desc.add_options()("half-length-mm",
                         po::value(&half_length)->default_value(half_length),
                         "Half length of plane [mm]");
    m_desc.add_options()("align-vector",
                         po::value(&align_vector)
                             ->value_name("X:Y:Z")
                             ->default_value(align_vector),
                         "Vector for plane placement");
}

void telescope_detector::read(const po::variables_map &) {

    thickness *= detray::unit<float>::mm;
    spacing *= detray::unit<float>::mm;
    smearing *= detray::unit<float>::um;
    half_length *= detray::unit<float>::mm;
}

std::unique_ptr<configuration_printable> telescope_detector::as_printable()
    const {
    std::unique_ptr<configuration_printable> cat =
        std::make_unique<configuration_category>("Telescope detector options");

    dynamic_cast<configuration_category &>(*cat).add_child(
        std::make_unique<configuration_kv_pair>("Empty material",
                                                empty_material ? "yes" : "no"));
    dynamic_cast<configuration_category &>(*cat).add_child(
        std::make_unique<configuration_kv_pair>("Number of planes",
                                                std::to_string(n_planes)));
    dynamic_cast<configuration_category &>(*cat).add_child(
        std::make_unique<configuration_kv_pair>(
            "Slab thickness",
            std::to_string(thickness / detray::unit<float>::mm) + " mm"));
    dynamic_cast<configuration_category &>(*cat).add_child(
        std::make_unique<configuration_kv_pair>(
            "Spacing",
            std::to_string(spacing / detray::unit<float>::mm) + " mm"));
    dynamic_cast<configuration_category &>(*cat).add_child(
        std::make_unique<configuration_kv_pair>(
            "Smearing",
            std::to_string(thickness / detray::unit<float>::um) + " um"));
    dynamic_cast<configuration_category &>(*cat).add_child(
        std::make_unique<configuration_kv_pair>(
            "Half length",
            std::to_string(half_length / detray::unit<float>::mm) + " mm"));
    std::stringstream align_ss;
    align_ss << align_vector;
    dynamic_cast<configuration_category &>(*cat).add_child(
        std::make_unique<configuration_kv_pair>("Alignment axis",
                                                align_ss.str()));

    return cat;
}

}  // namespace traccc::opts
