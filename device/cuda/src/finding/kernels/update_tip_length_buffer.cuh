/** traccc library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/vector.hpp>

#include "traccc/definitions/common.hpp"
#include "traccc/device/array_insertion_mutex.hpp"
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/finding/candidate_link.hpp"
#include "traccc/finding/finding_config.hpp"
#include "traccc/utils/prob.hpp"

namespace traccc::cuda::kernels {

__global__ void update_tip_length_buffer(
    const vecmem::data::vector_view<const unsigned int> old_tip_length_view,
    vecmem::data::vector_view<unsigned int> new_tip_length_view,
    const vecmem::data::vector_view<const unsigned int> measurement_votes_view,
    unsigned int* tip_to_output_map, unsigned int* tip_to_output_map_idx,
    float min_measurement_voting_fraction);

}  // namespace traccc::cuda::kernels
