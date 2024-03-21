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

/// Convenience namespace shorthand
namespace po = boost::program_options;

/// Description of this option group
static const char* description = "Detector Options";

detector::detector(po::options_description& desc) : m_desc{description} {

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
    desc.add(m_desc);
}

void detector::read(const po::variables_map&) {}

std::ostream& operator<<(std::ostream& out, const detector& opt) {

    out << ">>> " << description << " <<<\n"
        << "  Detector file       : " << opt.detector_file << "\n"
        << "  Material file       : " << opt.material_file << "\n"
        << "  Surface rid file    : " << opt.grid_file << "\n"
        << "  Use detray::detector: "
        << (opt.use_detray_detector ? "yes" : "no") << "\n"
        << "  Digitization file   : " << opt.digitization_file;
    return out;
}

}  // namespace traccc::opts
