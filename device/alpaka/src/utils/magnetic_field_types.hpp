/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/bfield/magnetic_field_types.hpp"
#include <covfie/hip/backend/primitive/hip_device_array.hpp>

namespace traccc::alpaka {

/// Inhomogeneous B-field backend type for Alpaka

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED)
template <typename scalar_t>
using inhom_bfield_backend_t =
    covfie::backend::affine<covfie::backend::linear<covfie::backend::clamp<
        covfie::backend::strided<covfie::vector::vector_d<std::size_t, 3>,
                                 covfie::backend::hip_device_array<
                                     covfie::vector::vector_d<scalar_t, 3>>>>>>;
// Test that the type is a valid backend for a field
static_assert(covfie::concepts::field_backend<inhom_bfield_backend_t<float>>,
              "hip::inhom_bfield_backend_t is not a valid field backend type");

/// @brief the standard list of Alpaka bfield types to support
template <typename scalar_t>
using bfield_type_list = std::tuple<const_bfield_backend_t<scalar_t>,
                                    host::inhom_bfield_backend_t<scalar_t>>;
#endif

}  // namespace traccc::alpaka
