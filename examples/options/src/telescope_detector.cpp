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

/// Description of this option group
static const char* description = "Telescope Detector Options";

telescope_detector::telescope_detector(po::options_description& desc)
    : m_desc{description} {

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
    desc.add(m_desc);
}

void telescope_detector::read(const po::variables_map&) {

    thickness *= detray::unit<float>::mm;
    spacing *= detray::unit<float>::mm;
    smearing *= detray::unit<float>::um;
    half_length *= detray::unit<float>::mm;
}

std::ostream& operator<<(std::ostream& out, const telescope_detector& opt) {

    out << ">>> " << description << " <<<\n"
        << "  Empty material   : " << (opt.empty_material ? "true" : "false")
        << "\n"
        << "  Number of planes : " << opt.n_planes << "\n"
        << "  Slab thickness   : " << opt.thickness / detray::unit<float>::mm
        << " [mm]\n"
        << "  Spacing          : " << opt.spacing / detray::unit<float>::mm
        << " [mm]\n"
        << "  Smearing         : " << opt.smearing / detray::unit<float>::um
        << " [um]\n"
        << "  Half length      : " << opt.half_length / detray::unit<float>::mm
        << " [mm]";
    return out;
}

}  // namespace traccc::opts
