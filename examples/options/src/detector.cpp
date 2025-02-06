/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/options/detector.hpp"

#include "details/configuration_category.hpp"
#include "details/configuration_value.hpp"

// System include(s).
#include <format>

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

std::unique_ptr<configuration_printable> detector::as_printable() const {

    auto result = std::make_unique<configuration_category>(m_description);

    result->add_child(
        std::make_unique<configuration_value>("Detector file", detector_file));
    result->add_child(
        std::make_unique<configuration_value>("Material file", material_file));
    result->add_child(
        std::make_unique<configuration_value>("Surface grid file", grid_file));
    result->add_child(std::make_unique<configuration_value>(
        "Use detray detector", std::format("{}", use_detray_detector)));
    result->add_child(std::make_unique<configuration_value>("Digitization file",
                                                            digitization_file));

    return result;
}

}  // namespace traccc::opts
