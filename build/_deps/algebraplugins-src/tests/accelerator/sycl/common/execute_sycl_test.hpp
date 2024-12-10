/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2020-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// SYCL include(s).
#include <sycl/sycl.hpp>

// System include(s).
#include <cstddef>

/// Execute a test functor using SYCL, on @c arraySizes threads
template <class FUNCTOR, class... ARGS>
void execute_sycl_test(::sycl::queue& queue, std::size_t arraySizes,
                       ARGS... args) {

  // Submit a kernel that would run the specified functor.
  queue
      .submit([&](::sycl::handler& h) {
        // Use parallel_for without specifying a "kernel class" explicitly.
        // Unfortunately the FUNCTOR class is too complicated, and DPC++ dies
        // on it. While providing a unique simple class for every template
        // specialisation is also pretty impossible. :-(
        h.parallel_for(::sycl::range<1>(arraySizes), [=](::sycl::item<1> id) {
          // Find the current index that we need to
          // process.
          const std::size_t i = id[0];
          if (i >= arraySizes) {
            return;
          }
          // Execute the test functor for this index.
          FUNCTOR()(i, args...);
        });
      })
      .wait_and_throw();
}
