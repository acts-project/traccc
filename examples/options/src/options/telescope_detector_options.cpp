/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/options/telescope_detector_options.hpp"

// Detray include(s).
#include "detray/definitions/units.hpp"

// System include(s).
#include <iostream>

namespace traccc {

/// Convenience namespace shorthand
namespace po = boost::program_options;

telescope_detector_options::telescope_detector_options(
    po::options_description& desc) {

    desc.add_options()("empty-material", po::bool_switch(&empty_material),
                       "Build detector without materials");
    desc.add_options()("n-planes",
                       po::value(&n_planes)->default_value(n_planes),
                       "Number of planes");
    desc.add_options()("thickness-mm",
                       po::value(&thickness)->default_value(thickness),
                       "Slab thickness in [mm]");
    desc.add_options()("spacing", po::value(&spacing)->default_value(spacing),
                       "Space between planes in [mm]");
    desc.add_options()("smearing-um",
                       po::value(&smearing)->default_value(smearing),
                       "Measurement smearing in [um]");
    desc.add_options()("half-length-mm",
                       po::value(&half_length)->default_value(half_length),
                       "Half length of plane [mm]");
}

void telescope_detector_options::read(const po::variables_map&) {

    thickness *= detray::unit<float>::mm;
    spacing *= detray::unit<float>::mm;
    smearing *= detray::unit<float>::um;
    half_length *= detray::unit<float>::mm;
}

std::ostream& operator<<(std::ostream& out,
                         const telescope_detector_options& opt) {

    out << ">>> Telescope detector options <<<\n"
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

}  // namespace traccc
