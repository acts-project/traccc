/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Detray include(s)
#include "detray/builders/grid_factory.hpp"
#include "detray/definitions/detail/containers.hpp"
#include "detray/materials/material_slab.hpp"
#include "detray/utils/grid/detail/axis_helpers.hpp"
#include "detray/utils/grid/detail/concepts.hpp"
#include "detray/utils/grid/detail/grid_bins.hpp"
#include "detray/utils/grid/grid.hpp"
#include "detray/utils/grid/serializers.hpp"

// System include(s)
#include <type_traits>

namespace detray {

/// Definition of binned material
template <typename shape, typename scalar_t,
          typename container_t = host_container_types, bool owning = false>
using material_map = grid<axes<shape>, bins::single<material_slab<scalar_t>>,
                          simple_serializer, container_t, owning>;

/// How to build material maps of various shapes
// TODO: Move to material_map_builder once available
template <typename scalar_t = detray::scalar>
using material_grid_factory =
    grid_factory<bins::single<material_slab<scalar_t>>, simple_serializer>;

}  // namespace detray
