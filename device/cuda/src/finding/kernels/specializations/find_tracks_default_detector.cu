/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "find_tracks_src.cuh"

namespace traccc::cuda::kernels {
template __global__ void find_tracks<traccc::default_detector::device>(
    const finding_config cfg,
    device::find_tracks_payload<traccc::default_detector::device> payload);
}
