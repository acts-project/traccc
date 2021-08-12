/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// array
#if defined(ALGEBRA_array)
#include "plugins/algebra/array_definitions.hpp"

namespace traccc {
template <typename T, std::size_t N>
using array = darray<T, N>;
using transform3 = algebra::array::transform3;
}  // namespace traccc

// eigen
#elif defined(ALGEBRA_eigen)
#include "plugins/algebra/eigen_definitions.hpp"

namespace traccc {
template <typename T, std::size_t N>
using array = Eigen::Matrix<T, N, 1>;
using transform3 = algebra::eigen::transform3;
}  // namespace traccc

// smatrix
#elif defined(ALGEBRA_smatrix)
#include "plugins/algebra/smatrix_definitions.hpp"

namespace traccc {
using transform3 = algebra::smatrix::transform3;
}

// vc_array
#elif defined(ALGEBRA_vc)
#include "plugins/algebra/vc_array_definitions.hpp"

namespace traccc {
using transform3 = algebra::vc_array::transform3;
}

#endif
