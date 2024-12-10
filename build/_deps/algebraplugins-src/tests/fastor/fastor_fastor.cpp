/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "algebra/fastor_fastor.hpp"

// Test include(s).
#include "test_host_basics.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <cstddef>  // for the std::size_t data type
#include <string>

/// Struct providing a readable name for the test
struct test_specialisation_name {
  template <typename T>
  static std::string GetName(int i) {
    switch (i) {
      case 0:
        return "fastor_fastor<float>";
      case 1:
        return "fastor_fastor<double>";
      default:
        return "unknown";
    }
  }
};

// Instantiate the test(s).
typedef testing::Types<
    test_types<float, algebra::fastor::point2<float>,
               algebra::fastor::point3<float>, algebra::fastor::vector2<float>,
               algebra::fastor::vector3<float>,
               algebra::fastor::transform3<float>, std::size_t,
               algebra::fastor::matrix_type, algebra::matrix::actor<float>>,
    test_types<double, algebra::fastor::point2<double>,
               algebra::fastor::point3<double>,
               algebra::fastor::vector2<double>,
               algebra::fastor::vector3<double>,
               algebra::fastor::transform3<double>, std::size_t,
               algebra::fastor::matrix_type, algebra::matrix::actor<double>>>
    fastor_fastor_types;
INSTANTIATE_TYPED_TEST_SUITE_P(algebra_plugins, test_host_basics,
                               fastor_fastor_types, test_specialisation_name);
