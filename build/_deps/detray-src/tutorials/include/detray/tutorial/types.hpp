/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

/// This file contains the definitions of the linear algebra types to be used in
/// the tutorials, which in this case will be the std::array based plugin.
/// The includes for the different algebra plugins can be found in
/// @c detray/plugins/algebra/

// Detray core include(s).
#include "detray/definitions/detail/algebra.hpp"
#include "detray/plugins/algebra/array_definitions.hpp"

namespace detray::tutorial {

using algebra_t = detray::cmath<detray::scalar>;
using transform3 = detray::dtransform3D<algebra_t>;
using point2 = detray::dpoint2D<algebra_t>;
using point3 = detray::dpoint3D<algebra_t>;
using vector3 = detray::dvector3D<algebra_t>;

}  // namespace detray::tutorial
