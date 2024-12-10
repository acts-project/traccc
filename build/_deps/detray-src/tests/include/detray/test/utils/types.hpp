/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2020-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Detray core include(s).
#include "detray/definitions/detail/algebra.hpp"

namespace detray::test {

using algebra = ALGEBRA_PLUGIN<detray::scalar>;

using scalar = dscalar<algebra>;
using point2 = dpoint2D<algebra>;
using point3 = dpoint3D<algebra>;
using vector3 = dvector3D<algebra>;
using transform3 = dtransform3D<algebra>;
using matrix_operator = dmatrix_operator<algebra>;
template <std::size_t ROWS, std::size_t COLS>
using matrix = dmatrix<algebra, ROWS, COLS>;

#if DETRAY_ALGEBRA_ARRAY
static constexpr char filenames[] = "array-";
#elif DETRAY_ALGEBRA_EIGEN
static constexpr char filenames[] = "eigen-";
#elif DETRAY_ALGEBRA_SMATRIX
static constexpr char filenames[] = "smatrix-";
#elif DETRAY_ALGEBRA_VC_AOS
static constexpr char filenames[] = "vc-";
#endif

}  // namespace detray::test
