/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// array
#if defined(ALGEBRA_ARRAY)
#include "plugins/algebra/array_definitions.hpp"

namespace traccc {
namespace array {

template <typename T, std::size_t N>
using array = darray<T, N>;
using transform3 = algebra::array::transform3;

}  // namespace array
}  // namespace traccc

#endif

// eigen
#if defined(ALGEBRA_EIGEN)
#include "plugins/algebra/eigen_definitions.hpp"

namespace traccc {
namespace eigen {

template <typename T, std::size_t N>
using array = Eigen::Matrix<T, N, 1>;
using transform3 = algebra::eigen::transform3;

}  // namespace eigen
}  // namespace traccc

#endif

// smatrix
#if defined(ALGEBRA_SMATRIX)
#include "plugins/algebra/smatrix_definitions.hpp"

namespace traccc {
namespace smatrix {}  // namespace smatrix
}  // namespace traccc

#endif

// vc_array
#if defined(ALGEBRA_VC)
#include "plugins/algebra/vc_array_definitions.hpp"

namespace traccc {
namespace vc {}  // namespace vc
}  // namespace traccc

#endif
