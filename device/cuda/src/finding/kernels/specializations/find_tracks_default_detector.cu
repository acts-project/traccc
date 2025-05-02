/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "find_tracks_src.cuh"

// Project include(s).
#include "traccc/geometry/detector.hpp"

namespace traccc::cuda {
template void find_tracks<traccc::default_detector::device>(
    const dim3& grid_size, const dim3& block_size, std::size_t shared_mem_size,
    const cudaStream_t& stream, const finding_config cfg,
    device::find_tracks_payload<traccc::default_detector::device> payload);
}  // namespace traccc::cuda
