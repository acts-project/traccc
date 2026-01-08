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

__global__ void gather_best_tips_per_measurement(
    const vecmem::data::vector_view<const unsigned int> tips_view,
    const vecmem::data::vector_view<const candidate_link> links_view,
    const edm::measurement_collection<default_algebra>::const_view
        measurements_view,
    vecmem::data::vector_view<unsigned long long int> insertion_mutex_view,
    vecmem::data::vector_view<unsigned int> tip_index_view,
    vecmem::data::vector_view<scalar> tip_pval_view,
    const unsigned int max_num_tracks_per_measurement);

}
