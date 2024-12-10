/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "algebra/eigen_eigen.hpp"

// System include(s)
#include <algorithm>
#include <random>

namespace algebra {

/// Fill an @c Eigen3 based vector with random values
template <typename vector_t>
inline void fill_random_vec(std::vector<vector_t> &collection) {

  auto rand_obj = []() { return vector_t::Random(); };

  collection.resize(collection.capacity());
  std::generate(collection.begin(), collection.end(), rand_obj);
}

/// Fill a @c Eigen3 based transform3 with random values
template <typename transform3_t>
inline void fill_random_trf(std::vector<transform3_t> &collection) {

  using vector_t = typename transform3_t::vector3;

  auto rand_obj = []() {
    vector_t x_axis, z_axis, t;

    x_axis = vector::normalize(vector_t::Random());
    z_axis = vector_t::Random();
    t = vector::normalize(vector_t::Random());

    // Gram-Schmidt projection
    typename transform3_t::scalar_type coeff =
        vector::dot(x_axis, z_axis) / getter::norm(x_axis);
    z_axis = x_axis - coeff * z_axis;

    return transform3_t{t, x_axis, vector::normalize(z_axis)};
  };

  collection.resize(collection.capacity());
  std::generate(collection.begin(), collection.end(), rand_obj);
}

}  // namespace algebra