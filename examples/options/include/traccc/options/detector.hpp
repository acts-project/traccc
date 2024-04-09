/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
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
    std::string detector_file = "tml_detector/trackml-detector.csv";
    /// The file containing the material description
    std::string material_file;
    /// The file containing the surface grid description
    std::string grid_file;
    /// Use detray::detector for the geometry handling
    bool use_detray_detector = false;

    /// The digitization configuration file
    std::string digitization_file =
        "tml_detector/default-geometric-config-generic.json";

    /// @}

    /// Constructor
    detector();

    private:
    /// Print the specific options of this class
    std::ostream& print_impl(std::ostream& out) const override;

};  // struct detector

}  // namespace traccc::opts
