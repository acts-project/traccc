/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2020-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// System include(s).
#include <cstddef>

/// Execute a test functor on the host
template <class FUNCTOR, class... ARGS>
void execute_host_test(std::size_t arraySizes, ARGS... args) {

  // Instantiate the functor.
  auto functor = FUNCTOR();

  // Execute the functor on all elements of the array(s).
  for (std::size_t i = 0; i < arraySizes; ++i) {
    functor(i, args...);
  }
}
