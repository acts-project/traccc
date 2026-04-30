/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/cuda/finding/combinatorial_kalman_filter_algorithm.hpp"

namespace traccc::cuda {

combinatorial_kalman_filter_algorithm::combinatorial_kalman_filter_algorithm(
    const finding_config& config, const traccc::memory_resource& mr,
    vecmem::copy& copy, cuda::stream& str, std::unique_ptr<const Logger> logger)
    : device::combinatorial_kalman_filter_algorithm(config, mr, copy,
                                                    std::move(logger)),
      cuda::algorithm_base(str) {}

}  // namespace traccc::cuda
