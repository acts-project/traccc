/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "../../utils/utils.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/finding/device/make_barcode_sequence.hpp"

namespace traccc::alpaka {

struct MakeBarcodeSequenceKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc, device::make_barcode_sequence_payload payload) const {

        device::global_index_t globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0];
        device::make_barcode_sequence(globalThreadIdx, payload);
    }
};

}  // namespace traccc::alpaka
