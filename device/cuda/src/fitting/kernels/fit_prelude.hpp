/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <cuda_runtime.h>

#include "traccc/edm/track_candidate.hpp"
#include "traccc/edm/track_state.hpp"

namespace traccc::cuda {
void fit_prelude(
    const dim3& grid_size, const dim3& block_size, std::size_t shared_mem_size,
    const cudaStream_t& stream,
    vecmem::data::vector_view<const unsigned int> param_ids_view,
    track_candidate_container_types::const_view track_candidates_view,
    track_state_container_types::view track_states_view,
    vecmem::data::vector_view<unsigned int> param_liveness_view);
}
