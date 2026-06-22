/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/device/global_index.hpp"

// Project include(s).
#include "traccc/edm/track_container.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {

/// Payload for the @c traccc::device::fit_prelude function
struct fit_prelude_payload {

    /// Input track parameter IDs
    vecmem::data::vector_view<const unsigned int> track_indices;
    /// Input tracks
    edm::track_container<default_algebra>::const_view input_tracks;
    /// Output tracks
    edm::track_container<default_algebra>::view output_tracks;
    /// Output track liveness
    vecmem::data::vector_view<unsigned int> track_liveness;

};  // struct fit_prelude_payload

/// Function to prepare the fitting payloads for the fitting algorithm
TRACCC_HOST_DEVICE inline void fit_prelude(const global_index_t globalIndex,
                                           const fit_prelude_payload& payload);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/fitting/device/impl/fit_prelude.ipp"
