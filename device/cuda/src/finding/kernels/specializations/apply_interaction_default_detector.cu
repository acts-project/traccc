/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "apply_interaction_src.cuh"
#include "types.hpp"

namespace traccc::cuda {

template void apply_interaction<traccc::default_detector::device>(
    const dim3& grid_size, const dim3& block_size, std::size_t shared_mem_size,
    const cudaStream_t& stream, const finding_config,
    device::apply_interaction_payload<traccc::default_detector::device>);
}
