/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// detray core
#include <detray/definitions/indexing.hpp>
#include <detray/grids/axis.hpp>
#include <detray/grids/grid2.hpp>
#include <detray/grids/populator.hpp>
#include <detray/grids/serializer2.hpp>

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/jagged_device_vector.hpp>
#include <vecmem/containers/jagged_vector.hpp>
#include <vecmem/containers/vector.hpp>

namespace traccc::details {

/// @brief Type definitions for the spacepoint grid.
///
/// This struct contains all the type definitions that are needed to work with
/// the spacepoint grid.
///
struct spacepoint_grid_types {

    /// Spacepoint grid host type
    using host =
        detray::grid2<detray::attach_populator, detray::axis2::circular,
                      detray::axis2::regular, detray::serializer2,
                      vecmem::vector, vecmem::jagged_vector, detray::darray,
                      detray::dtuple, unsigned int, false>;

    /// Spacepoint grid (non-const) device type
    using device =
        detray::grid2<detray::attach_populator, detray::axis2::circular,
                      detray::axis2::regular, detray::serializer2,
                      vecmem::device_vector, vecmem::jagged_device_vector,
                      detray::darray, detray::dtuple, unsigned int, false>;
    /// Spacepoint grid (const) device type
    using const_device =
        detray::grid2<detray::attach_populator, detray::axis2::circular,
                      detray::axis2::regular, detray::serializer2,
                      vecmem::device_vector, vecmem::jagged_device_vector,
                      detray::darray, detray::dtuple, const unsigned int,
                      false>;

    /// Spacepoint grid (non-const) data type
    using data = detray::grid2_data<host>;
    /// Spacepoint grid (const) data type
    using const_data = detray::const_grid2_data<host>;

    /// Spacepoint grid (non-const) view type
    using view = detray::grid2_view<host>;
    /// Spacepoint grid (const) view type
    using const_view = detray::const_grid2_view<host>;

    /// Spacepoint grid buffer type
    using buffer = detray::grid2_buffer<host>;

};  // struct spacepoint_grid_types

}  // namespace traccc::details
