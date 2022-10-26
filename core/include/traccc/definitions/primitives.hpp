/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
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
#include "traccc/plugins/algebra/vc_definitions.hpp"
#elif ALGEBRA_PLUGINS_INCLUDE_VECMEM
#include "traccc/plugins/algebra/vecmem_definitions.hpp"
#endif

#include <Eigen/Core>

namespace traccc {

using geometry_id = uint64_t;
using event_id = uint64_t;
using channel_id = unsigned int;

using vector2 = __plugin::point2<traccc::scalar>;
using point2 = __plugin::point2<traccc::scalar>;
using variance2 = __plugin::point2<traccc::scalar>;
using point3 = __plugin::point3<traccc::scalar>;
using vector3 = __plugin::point3<traccc::scalar>;
using variance3 = __plugin::point3<traccc::scalar>;
using transform3 = __plugin::transform3<traccc::scalar>;

}  // namespace traccc
