/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/core/detail/multi_store.hpp"
#include "detray/definitions/detail/algebra.hpp"
#include "detray/geometry/mask.hpp"
#include "detray/geometry/shapes.hpp"

// Vecmem include(s)
#include <vecmem/containers/data/jagged_vector_buffer.hpp>
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/jagged_device_vector.hpp>
#include <vecmem/utils/cuda/copy.hpp>

namespace detray {

using algebra_t = ALGEBRA_PLUGIN<detray::scalar>;
using point3 = detray::dpoint3D<algebra_t>;
using transform3 = detray::dtransform3D<algebra_t>;
const int n_points = 1000;

using annulus = mask<annulus2D>;
using cylinder = mask<cylinder2D>;
using rectangle = mask<rectangle2D>;
using ring = mask<ring2D>;
using single = mask<single3D<>>;
using trapezoid = mask<trapezoid2D>;

/// Enumerate different mask types for convenience
enum class mask_ids : unsigned int {
    e_rectangle2 = 0u,
    e_trapezoid2 = 1u,
    e_ring2 = 2u,
    e_cylinder2 = 3u,
    e_single3 = 4u,
    e_annulus2 = 5u,
};

using host_store_type =
    regular_multi_store<mask_ids, empty_context, dtuple, vecmem::vector,
                        rectangle, trapezoid, ring, cylinder, single, annulus>;

using device_store_type =
    regular_multi_store<mask_ids, empty_context, dtuple, vecmem::device_vector,
                        rectangle, trapezoid, ring, cylinder, single, annulus>;

/// test function for mask store
void mask_test(typename host_store_type::view_type store_data,
               vecmem::data::vector_view<point3> input_point3_data,
               vecmem::data::jagged_vector_view<int> output_data);

}  // namespace detray
