/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/edm/track_parameters.hpp"
#include "traccc/finding/device/fill_sort_keys.hpp"

namespace traccc::cuda::kernels {

__global__ void fill_sort_keys(device::fill_sort_keys_payload payload);
}
