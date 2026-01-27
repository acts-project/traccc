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

__global__ void gather_measurement_votes(
    const vecmem::data::vector_view<const unsigned long long int>
        insertion_mutex_view,
    const vecmem::data::vector_view<const unsigned int> tip_index_view,
    vecmem::data::vector_view<unsigned int> votes_per_tip_view,
    const unsigned int max_num_tracks_per_measurement);

}  // namespace traccc::cuda::kernels
