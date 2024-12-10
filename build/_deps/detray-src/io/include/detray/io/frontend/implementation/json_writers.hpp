/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/io/common/geometry_writer.hpp"
#include "detray/io/common/homogeneous_material_writer.hpp"
#include "detray/io/common/material_map_writer.hpp"
#include "detray/io/common/surface_grid_writer.hpp"
#include "detray/io/frontend/detail/detector_components_writer.hpp"
#include "detray/io/frontend/detail/type_traits.hpp"
#include "detray/io/frontend/detector_writer_config.hpp"
#include "detray/io/json/json_writer.hpp"

namespace detray::io {

struct detector_writer_config;

namespace detail {

/// Infer the writers that are needed from the detector type @tparam detector_t
template <class detector_t>
void add_json_writers(detector_components_writer<detector_t>& writers,
                      const detray::io::detector_writer_config& cfg) {

    // Always needed
    using json_geometry_writer = json_writer<detector_t, geometry_writer>;

    writers.template add<json_geometry_writer>();

    // Find other writers, depending on the detector type
    if (cfg.write_material()) {
        // Simple material
        if constexpr (concepts::has_homogeneous_material<detector_t>) {
            using json_homogeneous_material_writer =
                json_writer<detector_t, homogeneous_material_writer>;

            writers.template add<json_homogeneous_material_writer>();
        }
        // Material maps
        if constexpr (concepts::has_material_maps<detector_t>) {
            using json_material_map_writer =
                json_writer<detector_t, material_map_writer>;

            writers.template add<json_material_map_writer>();
        }
    }
    // Navigation acceleration structures
    if constexpr (concepts::has_surface_grids<detector_t>) {
        using json_surface_grid_writer =
            json_writer<detector_t, surface_grid_writer>;

        if (cfg.write_grids()) {
            writers.template add<json_surface_grid_writer>();
        }
    }
}

}  // namespace detail

}  // namespace detray::io
