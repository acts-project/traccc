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

// Fixme: Need to utilize algebra plugin for vector and matrix
template <unsigned int kSize>
using traccc_vector = Eigen::Matrix<scalar, kSize, 1>;

template <unsigned int kRows, unsigned int kCols>
using traccc_matrix = Eigen::Matrix<scalar, kRows, kCols>;

template <unsigned int kSize>
using traccc_sym_matrix = Eigen::Matrix<scalar, kSize, kSize>;

}  // namespace traccc

namespace detray {

using scalar = traccc::scalar;

template <typename value_type, unsigned int kDIM>
using darray = traccc::darray<value_type, kDIM>;

template <typename value_type>
using dvector = traccc::dvector<value_type>;

template <typename value_type>
using djagged_vector = traccc::djagged_vector<value_type>;

template <typename key_type, typename value_type>
using dmap = traccc::dmap<key_type, value_type>;

template <class... types>
using dtuple = traccc::dtuple<types...>;

}  // namespace detray
