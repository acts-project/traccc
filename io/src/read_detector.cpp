/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/io/read_detector.hpp"

#include "traccc/io/utils.hpp"

// Detray include(s).
#include <detray/io/frontend/detector_reader.hpp>
#include <detray/io/frontend/detector_reader_config.hpp>

// System include(s).
#include <string>

namespace {

/// Common implementation for constructing a detector from a set of input files
template <typename detector_t>
void read_detector(detector_t& detector, vecmem::memory_resource& mr,
                   const std::string_view& geometry_file,
                   const std::string_view& material_file,
                   const std::string_view& grid_file) {

    // Set up the detector reader configuration.
    detray::io::detector_reader_config cfg;
    cfg.add_file(traccc::io::get_absolute_path(geometry_file));
    if (material_file.empty() == false) {
        cfg.add_file(traccc::io::get_absolute_path(material_file));
    }
    if (grid_file.empty() == false) {
        cfg.add_file(traccc::io::get_absolute_path(grid_file));
    }

    // Read the detector.
    auto det = detray::io::read_detector<detector_t>(mr, cfg);
    detector = std::move(det.first);
}

}  // namespace

namespace traccc::io {

void read_detector(default_detector::host& detector,
                   vecmem::memory_resource& mr,
                   const std::string_view& geometry_file,
                   const std::string_view& material_file,
                   const std::string_view& grid_file) {

    ::read_detector(detector, mr, geometry_file, material_file, grid_file);
}

}  // namespace traccc::io
