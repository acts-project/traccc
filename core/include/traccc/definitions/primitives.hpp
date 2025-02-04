/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#if ALGEBRA_PLUGINS_INCLUDE_ARRAY
#include "traccc/plugins/algebra/array_definitions.hpp"
#elif ALGEBRA_PLUGINS_INCLUDE_EIGEN
#include "traccc/plugins/algebra/eigen_definitions.hpp"
#elif ALGEBRA_PLUGINS_INCLUDE_SMATRIX
#include "traccc/plugins/algebra/smatrix_definitions.hpp"
#elif ALGEBRA_PLUGINS_INCLUDE_VC
#include "traccc/plugins/algebra/vc_aos_definitions.hpp"
#elif ALGEBRA_PLUGINS_INCLUDE_VECMEM
#include "traccc/plugins/algebra/vecmem_definitions.hpp"
#endif

// Detray include(s)
#include <detray/definitions/algebra.hpp>

// System include(s)
#include <cstdint>

namespace traccc {

using measurement_id = std::uint64_t;
using particle_id = std::uint64_t;
using geometry_id = std::uint64_t;
using channel_id = unsigned int;

// Default algebra type
using default_algebra = ALGEBRA_PLUGIN<traccc::scalar>;

using scalar = detray::dscalar<default_algebra>;
using point2 = detray::dpoint2D<default_algebra>;
using vector2 = point2;
using variance2 = point2;
using point3 = detray::dpoint3D<default_algebra>;
using vector3 = detray::dvector3D<default_algebra>;
using variance3 = point3;
using transform3 = detray::dtransform3D<default_algebra>;

namespace getter = detray::getter;
namespace vector = detray::vector;
namespace matrix = detray::matrix;

}  // namespace traccc
