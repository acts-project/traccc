/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../utils/bfield.cuh"
#include "kalman_fitting.cuh"
#include "traccc/cuda/fitting/kalman_fitting_algorithm.hpp"

namespace traccc::cuda {

kalman_fitting_algorithm::output_type kalman_fitting_algorithm::operator()(
    const telescope_detector::view& det, const bfield& field,
    const edm::track_candidate_container<default_algebra>::const_view&
        track_candidates) const {

    // Run the track fitting.
    return bfield_visitor<cuda::bfield_type_list<scalar>>(
        field, [&]<typename bfield_view_t>(const bfield_view_t& bfield) {
            return details::kalman_fitting<telescope_detector::device>(
                det, bfield, track_candidates, m_config, m_mr, m_copy.get(),
                m_stream, m_warp_size);
        });
}

}  // namespace traccc::cuda
