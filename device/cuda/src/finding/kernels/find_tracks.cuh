/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/finding/device/find_tracks.hpp"
#include "traccc/finding/finding_config.hpp"

namespace traccc::cuda {

template <typename detector_t>
void find_tracks(const dim3& grid_size, const dim3& block_size,
                 std::size_t shared_mem_size, const cudaStream_t& stream,
                 const finding_config cfg,
                 device::find_tracks_payload<detector_t> payload);
}  // namespace traccc::cuda
