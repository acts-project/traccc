/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/options/details/interface.hpp"

// System include(s).
#include <string>

namespace traccc::opts {

/// Options for the detector description
class detector : public interface {

    public:
    /// @name Options
    /// @{

    /// The file containing the detector description
    std::string detector_file =
        "geometries/odd/odd-detray_geometry_detray.json";
    /// The file containing the material description
    std::string material_file =
        "geometries/odd/odd-detray_material_detray.json";
    /// The file containing the surface grid description
    std::string grid_file =
        "geometries/odd/odd-detray_surface_grids_detray.json";
    /// Use detray::detector for the geometry handling
    bool use_detray_detector = true;

    /// The digitization configuration file
    std::string digitization_file =
        "geometries/odd/odd-digi-geometric-config.json";

    /// @}

    /// Constructor
    detector();

    std::unique_ptr<configuration_printable> as_printable() const override;
};  // struct detector

}  // namespace traccc::opts
