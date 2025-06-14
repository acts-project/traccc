/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <covfie/core/backend/primitive/constant.hpp>
#include <covfie/core/backend/transformer/affine.hpp>
#include <covfie/core/backend/transformer/linear.hpp>
#include <covfie/core/backend/transformer/strided.hpp>
#include <covfie/core/field.hpp>
#include <covfie/core/vector.hpp>

// Covfie include(s)
#include <covfie/cuda/backend/primitive/cuda_device_array.hpp>

namespace traccc::cuda {

template <typename scalar_t>
using inhom_bfield_backend_t = covfie::backend::affine<covfie::backend::linear<
    covfie::backend::strided<covfie::vector::vector_d<std::size_t, 3>,
                             covfie::backend::cuda_device_array<
                                 covfie::vector::vector_d<scalar_t, 3>>>>>;

/// @brief Function that reads the first 4 bytes of a potential bfield file and
/// checks that it contains data for a covfie field
inline bool check_covfie_file(const std::string& file_name);

/// @brief function that reads a covfie field from file
template <typename bfield_t>
inline bfield_t read_bfield(const std::string& file_name);

template <typename scalar_t>
inline inhom_bfield_backend_t<scalar_t> create_inhom_bfield();

template <typename scalar_t>
inline inhom_bfield_backend_t<scalar_t> create_inhom_bfield(
    const std::string& file_name);

}  // namespace traccc::cuda
