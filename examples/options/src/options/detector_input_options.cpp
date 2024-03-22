/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/options/detector_input_options.hpp"

// System include(s).
#include <iostream>

namespace traccc {

/// Convenience namespace shorthand
namespace po = boost::program_options;

traccc::detector_input_options::detector_input_options(
    po::options_description& desc) {

    desc.add_options()("detector-file",
                       po::value(&detector_file)->default_value(detector_file),
                       "specify detector file");
    desc.add_options()("material-file",
                       po::value(&material_file)->default_value(material_file),
                       "specify material file");
    desc.add_options()("grid-file",
                       po::value(&grid_file)->default_value(grid_file),
                       "specify surface grid file");
    desc.add_options()("use-detray-detector",
                       po::bool_switch(&use_detray_detector),
                       "Use detray::detector for the geometry handling");
}

void traccc::detector_input_options::read(const po::variables_map&) {}

std::ostream& operator<<(std::ostream& out, const detector_input_options& opt) {

    out << ">>> Detector options <<<\n"
        << "  Detector file        : " << opt.detector_file << "\n"
        << "  Material file        : " << opt.material_file << "\n"
        << "  Grid file            : " << opt.grid_file << "\n"
        << "  Use detray::detector : "
        << (opt.use_detray_detector ? "yes" : "no") << "\n";
    return out;
}

}  // namespace traccc
