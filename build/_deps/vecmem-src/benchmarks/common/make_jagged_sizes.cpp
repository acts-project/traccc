/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "make_jagged_sizes.hpp"

// System include(s).
#include <algorithm>
#include <cassert>
#include <limits>
#include <random>

namespace vecmem::benchmark {

std::vector<std::size_t> make_jagged_sizes(int64_t outerSize,
                                           int64_t maxInnerSize) {

    // Some security checks.
    assert((outerSize >= 0) &&
           (outerSize < std::numeric_limits<int32_t>::max()));
    assert((maxInnerSize >= 0) &&
           (maxInnerSize < std::numeric_limits<int32_t>::max()));

    // Set up a simple random number generator for the inner vector sizes.
    std::default_random_engine eng;
    eng.seed(static_cast<std::default_random_engine::result_type>(
        outerSize + maxInnerSize));
    std::uniform_int_distribution<std::size_t> gen(
        0, static_cast<std::size_t>(maxInnerSize));

    // Generate the result vector.
    std::vector<std::size_t> result(static_cast<std::size_t>(outerSize));
    std::generate(result.begin(), result.end(),
                  [&eng, &gen]() { return gen(eng); });

    // Give it to the user.
    return result;
}

}  // namespace vecmem::benchmark
