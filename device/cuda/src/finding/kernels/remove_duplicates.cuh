/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/finding/device/remove_duplicates.hpp"

namespace traccc::cuda::kernels {

__global__ void remove_duplicates(const finding_config cfg,
                                  device::remove_duplicates_payload payload);
}
