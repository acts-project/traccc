/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "build_tracks.cuh"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_candidate.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/finding/candidate_link.hpp"
#include "traccc/finding/device/build_tracks.hpp"
#include "traccc/finding/finding_config.hpp"

namespace traccc::cuda::kernels {

__global__ void build_tracks(
    const finding_config cfg,
    measurement_collection_types::const_view measurements_view,
    bound_track_parameters_collection_types::const_view seeds_view,
    vecmem::data::jagged_vector_view<const candidate_link> links_view,
    vecmem::data::vector_view<const typename candidate_link::link_index_type>
        tips_view,
    track_candidate_container_types::view track_candidates_view,
    vecmem::data::vector_view<unsigned int> valid_indices_view,
    unsigned int* n_valid_tracks) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    device::build_tracks(
        gid, cfg,
        {measurements_view, seeds_view, links_view, tips_view,
         track_candidates_view, valid_indices_view, *n_valid_tracks});
}
}  // namespace traccc::cuda::kernels
