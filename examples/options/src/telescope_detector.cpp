/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/options/telescope_detector.hpp"

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

void telescope_detector::read(const po::variables_map&) {

    thickness *= detray::unit<float>::mm;
    spacing *= detray::unit<float>::mm;
    smearing *= detray::unit<float>::um;
    half_length *= detray::unit<float>::mm;
}

std::ostream& telescope_detector::print_impl(std::ostream& out) const {

    out << "  Empty material  : " << (empty_material ? "true" : "false") << "\n"
        << "  Number of planes: " << n_planes << "\n"
        << "  Slab thickness  : " << thickness / detray::unit<float>::mm
        << " [mm]\n"
        << "  Spacing         : " << spacing / detray::unit<float>::mm
        << " [mm]\n"
        << "  Smearing        : " << smearing / detray::unit<float>::um
        << " [um]\n"
        << "  Half length     : " << half_length / detray::unit<float>::mm
        << " [mm]"
        << "  Align axis      : " << align_vector << " \n";
    return out;
}

}  // namespace traccc::opts
