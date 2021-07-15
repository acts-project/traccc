/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// traccc include(s).
#include "definitions/algebra.hpp"
#include "definitions/primitives.hpp"
#include "edm/container.hpp"
#include "geometry/pixel_segmentation.hpp"

// VecMem include(s).
#include <vecmem/containers/data/jagged_vector_buffer.hpp>
#include <vecmem/containers/data/vector_buffer.hpp>
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/jagged_device_vector.hpp>
#include <vecmem/containers/jagged_vector.hpp>
#include <vecmem/containers/vector.hpp>

// System include(s).
#include <limits>

namespace traccc {

/// @name Types to use in algorithmic code
/// @{

/// A cell definition:
///
/// maximum two channel identifiers
/// and one activiation value, such as a time stamp
struct cell {
    channel_id channel0 = 0;
    channel_id channel1 = 0;
    scalar activation = 0.;
    scalar time = 0.;
};

/// Container of cells belonging to one detector module
template <template <typename> class vector_t>
using cell_collection = vector_t<cell>;

/// Convenience declaration for the cell collection type to use in host code
using host_cell_collection = cell_collection<vecmem::vector>;
/// Convenience declaration for the cell collection type to use in device code
using device_cell_collection = cell_collection<vecmem::device_vector>;

/// Header information for all of the cells in a specific detector module
///
/// It is handled separately from the list of all of the cells belonging to
/// the detector module, to be able to lay out the data in memory in a way
/// that is more friendly towards accelerators.
///
struct cell_module {

    event_id event = 0;
    geometry_id module = 0;
    transform3 placement = transform3{};

    channel_id range0[2] = {std::numeric_limits<channel_id>::max(), 0};
    channel_id range1[2] = {std::numeric_limits<channel_id>::max(), 0};

    pixel_segmentation pixel;
};  // struct cell_module

/// Convenience declaration for the cell container type to use in host code
using host_cell_container = host_container<cell_module, cell>;

/// Convenience declaration for the cell container type to use in device code
using device_cell_container = device_container<cell_module, cell>;

/// Convenience declaration for the cell container data type to use in host code
using cell_container_data = container_data<cell_module, cell>;

/// Convenience declaration for the cell container buffer type to use in host
/// code
using cell_container_buffer = container_buffer<cell_module, cell>;

/// Convenience declaration for the cell container view type to use in host code
using cell_container_view = container_view<cell_module, cell>;

}  // namespace traccc
