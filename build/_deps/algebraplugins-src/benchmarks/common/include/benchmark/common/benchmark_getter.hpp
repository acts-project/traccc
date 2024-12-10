/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "benchmark_vector.hpp"

namespace algebra::bench_op {

// Macro for declaring the predefined materials (with Density effect data)
#define ALGEBRA_PLUGINS_BENCH_GETTER(GETTER_NAME)       \
  struct GETTER_NAME {                                  \
    inline static const std::string name{#GETTER_NAME}; \
    template <typename vector_t>                        \
    auto operator()(const vector_t &a) const {          \
      return algebra::getter::GETTER_NAME(a);           \
    }                                                   \
  };

// Functions to be benchmarked
ALGEBRA_PLUGINS_BENCH_GETTER(phi)
ALGEBRA_PLUGINS_BENCH_GETTER(theta)
ALGEBRA_PLUGINS_BENCH_GETTER(perp)
ALGEBRA_PLUGINS_BENCH_GETTER(norm)
ALGEBRA_PLUGINS_BENCH_GETTER(eta)

}  // namespace algebra::bench_op
