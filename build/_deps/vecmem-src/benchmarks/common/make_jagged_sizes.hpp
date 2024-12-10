/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// System include(s).
#include <cstddef>
#include <cstdint>
#include <vector>

namespace vecmem::benchmark {

/// Helper function for generating the sizes for a jagged vector (buffer)
///
/// It implements a pretty simple thing, but since this is used in multiple
/// places, it made sense to put it into a central location.
///
/// @param outerSize The fixed "outer size" of the jagged vector (buffer)
/// @param maxInnerSize The maximum for the random "inner sizes" of the
///                     resulting vector (buffer)
/// @return A vector of sizes corresponding to the received parameters
///
std::vector<std::size_t> make_jagged_sizes(int64_t outerSize,
                                           int64_t maxInnerSize);

}  // namespace vecmem::benchmark
