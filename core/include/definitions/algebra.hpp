/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// array plugin
#ifdef ALGEBRA_PLUGIN_INCLUDE_ARRAY
#include "plugins/algebra/array_definitions.hpp"

namespace traccc {

template <typename T, std::size_t N>
using array = std::array<T, N>;
using transform3 = algebra::std_array::transform3;

}  // namespace traccc

#endif

// eigen plugin
#ifdef ALGEBRA_PLUGIN_INCLUDE_EIGEN
#include "plugins/algebra/eigen_definitions.hpp"

namespace traccc {

template <typename T, std::size_t N>
using array = Eigen::Matrix<T, N, 1>;
using transform3 = algebra::eigen::transform3;

}  // namespace traccc

#endif

// vecmem plugin
#ifdef ALGEBRA_PLUGIN_INCLUDE_VECMEM
#include "plugins/algebra/vecmem_definitions.hpp"

namespace traccc {

template <typename T, std::size_t N>
using array = vecmem::static_array<T, N>;
using transform3 = algebra::array::transform3;

}  // namespace traccc

#endif
