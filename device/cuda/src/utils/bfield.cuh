/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Covfie include(s).
#include <covfie/core/backend/transformer/affine.hpp>
#include <covfie/core/field.hpp>
#include <covfie/core/vector.hpp>
#include <covfie/cuda/backend/primitive/cuda_device_array.hpp>
#include <covfie/cuda/backend/primitive/cuda_texture.hpp>

namespace traccc::cuda {

/// Inhomogeneous B-field backend type for CUDA
template <typename scalar_t>
using inhom_bfield_backend_t = covfie::backend::affine<
    covfie::backend::cuda_texture<covfie::vector::vector_d<scalar_t, 3>,
                                  covfie::vector::vector_d<scalar_t, 3>>>;

}  // namespace traccc::cuda
