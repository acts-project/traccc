/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "edm/internal_spacepoint.hpp"

// detray core
#include "detray/grids/axis.hpp"
#include "detray/grids/grid2.hpp"
#include "detray/grids/populator.hpp"
#include "detray/grids/serializer2.hpp"
#include "detray/utils/indexing.hpp"

namespace traccc {

using sp_grid =
    detray::grid2<detray::attach_populator, detray::axis::circular,
                  detray::axis::regular, detray::serializer2, detray::dvector,
                  detray::djagged_vector, detray::darray, detray::dtuple,
                  internal_spacepoint<spacepoint>, false>;

using sp_grid_device = detray::grid2<
    detray::attach_populator, detray::axis::circular, detray::axis::regular,
    detray::serializer2, vecmem::device_vector, vecmem::jagged_device_vector,
    detray::darray, detray::dtuple, internal_spacepoint<spacepoint>, false>;

using sp_grid_data = detray::grid2_data<sp_grid>;
using sp_grid_view = detray::grid2_view<sp_grid>;
using sp_grid_buffer = detray::grid2_buffer<sp_grid>;

}  // namespace traccc