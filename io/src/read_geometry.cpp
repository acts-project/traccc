/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/io/read_geometry.hpp"

#include "traccc/geometry/detector.hpp"
#include "traccc/io/details/read_surfaces.hpp"
#include "traccc/io/utils.hpp"

// Detray include(s).
#include <detray/io/frontend/detector_reader.hpp>

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// System include(s).
#include <stdexcept>

namespace {

/// Helper function constructing @c traccc::geometry from Detray JSON
std::pair<traccc::geometry,
          std::unique_ptr<std::map<std::uint64_t, detray::geometry::barcode>>>
read_json_geometry(std::string_view filename) {

    // Memory resource used while reading the detector JSON.
    vecmem::host_memory_resource host_mr;

    // The result objects.
    traccc::geometry surface_transforms;
    std::unique_ptr<std::map<std::uint64_t, detray::geometry::barcode>>
        barcode_map;

    {
        // Construct a detector object.
        detray::io::detector_reader_config reader_cfg{};
        reader_cfg.add_file(std::string{filename});
        auto [detector, _] =
            detray::io::read_detector<traccc::default_detector::host>(
                host_mr, reader_cfg);

        // Construct an "old style geometry" from the detector object.
        surface_transforms = traccc::io::alt_read_geometry(detector);

        // Construct a map from Acts surface identifiers to Detray barcodes.
        barcode_map = std::make_unique<
            std::map<std::uint64_t, detray::geometry::barcode>>();
        for (const auto& surface : detector.surfaces()) {
            (*barcode_map)[surface.source] = surface.barcode();
        }
    }

    // Return the created objects.
    return {surface_transforms, std::move(barcode_map)};
}

}  // namespace

namespace traccc::io {

std::pair<geometry,
          std::unique_ptr<std::map<std::uint64_t, detray::geometry::barcode>>>
read_geometry(std::string_view filename, data_format format) {

    // Construct the full file name.
    const std::string full_filename = get_absolute_path(filename);

    // Decide how to read the file.
    switch (format) {
        case data_format::csv:
            return {geometry{details::read_surfaces(full_filename, format)},
                    nullptr};
        case data_format::json:
            return ::read_json_geometry(full_filename);
        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

}  // namespace traccc::io
