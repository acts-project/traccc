/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "algebra/array_cmath.hpp"

// System include(s)
#include <algorithm>
#include <random>
#include <vector>

namespace algebra {

/// Fill an @c std::array based vector with random values
template <typename vector_t>
inline void fill_random_vec(std::vector<vector_t> &collection) {

  // Generate a vector of the right type with random values
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<typename vector_t::value_type> dist(0.f, 1.f);

  auto rand_obj = [&]() { return vector_t{dist(mt), dist(mt), dist(mt)}; };

  collection.resize(collection.capacity());
  std::generate(collection.begin(), collection.end(), rand_obj);
}

/// Fill a @c Vc::Vector based transform3 with random values
template <typename transform3_t>
inline void fill_random_trf(std::vector<transform3_t> &collection) {

  using vector_t = typename transform3_t::vector3;

  // Generate a random, but valid affine transformation
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<typename transform3_t::scalar_type> dist(0.f,
                                                                          1.f);

  auto rand_obj = [&]() {
    vector_t x_axis, z_axis, t;

    x_axis = vector::normalize(vector_t{dist(mt), dist(mt), dist(mt)});
    z_axis = {dist(mt), dist(mt), dist(mt)};
    t = vector::normalize(vector_t{dist(mt), dist(mt), dist(mt)});

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