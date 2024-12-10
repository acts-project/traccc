/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project includes(s)
#include "detray/core/detail/single_store.hpp"
#include "detray/definitions/detail/algebra.hpp"

// Vecmem include(s)
#include <vecmem/containers/device_vector.hpp>

namespace detray {

using algebra_t = ALGEBRA_PLUGIN<detray::scalar>;
using point3 = dpoint3D<algebra_t>;
using transform3 = dtransform3D<algebra_t>;

using host_transform_store_t = single_store<transform3, vecmem::vector>;

using device_transform_store_t =
    single_store<transform3, vecmem::device_vector>;

void transform_test(vecmem::data::vector_view<point3> input_data,
                    typename host_transform_store_t::view_type store_data,
                    vecmem::data::vector_view<point3> output_data,
                    std::size_t n_transforms);

}  // namespace detray
