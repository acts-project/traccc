/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "kalman_fitting.cuh"
#include "traccc/cuda/fitting/kalman_fitting_algorithm.hpp"

namespace traccc::cuda {

kalman_fitting_algorithm::output_type kalman_fitting_algorithm::operator()(
    const default_detector::view& det,
    const covfie::field<traccc::const_bfield_backend_t<
        default_detector::device::scalar_type>>::view_t& field,
    const edm::track_candidate_container<default_algebra>::const_view&
        track_candidates) const {

    // Run the track fitting.
    return details::kalman_fitting<default_detector::device>(
        det, field, track_candidates, m_config, m_mr, m_copy.get(), m_stream,
        m_warp_size);
}

}  // namespace traccc::cuda
