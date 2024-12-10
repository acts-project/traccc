/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/io/frontend/detail/detector_components_writer.hpp"
#include "detray/io/frontend/detector_writer_config.hpp"
#include "detray/io/frontend/implementation/json_writers.hpp"
#include "detray/io/utils/create_path.hpp"

// System include(s)
#include <filesystem>
#include <ios>

namespace detray::io {

/// @brief Writer function for detray detectors.
///
/// Based on both the given config/file format and the detector type,
/// the correct writers are being assembled and called.
/// Write @param det to file in the format @param format,
/// using @param names to name the components
template <class detector_t>
void write_detector(detector_t& det, const typename detector_t::name_map& names,
                    detector_writer_config& cfg) {
    // How to open the file
    const std::ios_base::openmode out_mode{std::ios_base::out |
                                           std::ios_base::binary};
    auto mode =
        cfg.replace_files() ? (out_mode | std::ios_base::trunc) : out_mode;

    const auto file_path = detray::io::create_path(cfg.path());

    if (det.material_store().all_empty()) {
        cfg.write_material(false);
    }

    // Get the writer
    io::detail::detector_components_writer<detector_t> writer{};
    if (cfg.format() == io::format::json) {
        detail::add_json_writers(writer, cfg);
    }

    if (cfg.compactify_json()) {
        std::cout << "WARNING: Compactifying json files is not yet implemented"
                  << std::endl;
    }

    writer.write(det, names, mode, file_path);
}

}  // namespace detray::io
