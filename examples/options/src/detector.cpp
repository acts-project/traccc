/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/options/detector.hpp"

// System include(s).
#include <iostream>

namespace traccc::opts {

detector::detector() : interface("Detector Options") {

    namespace po = boost::program_options;

    m_desc.add_options()(
        "detector-file",
        po::value(&detector_file)->default_value(detector_file),
        "Detector file");
    m_desc.add_options()(
        "material-file",
        po::value(&material_file)->default_value(material_file),
        "Material file");
    m_desc.add_options()("grid-file",
                         po::value(&grid_file)->default_value(grid_file),
                         "Surface grid file");
    m_desc.add_options()("use-detray-detector",
                         po::bool_switch(&use_detray_detector),
                         "Use detray::detector for the geometry handling");
    m_desc.add_options()(
        "digitization-file",
        po::value(&digitization_file)->default_value(digitization_file),
        "Digitization file");
}

std::ostream& detector::print_impl(std::ostream& out) const {

    out << "  Detector file       : " << detector_file << "\n"
        << "  Material file       : " << material_file << "\n"
        << "  Surface rid file    : " << grid_file << "\n"
        << "  Use detray::detector: " << (use_detray_detector ? "yes" : "no")
        << "\n"
        << "  Digitization file   : " << digitization_file;
    return out;
}

}  // namespace traccc::opts
