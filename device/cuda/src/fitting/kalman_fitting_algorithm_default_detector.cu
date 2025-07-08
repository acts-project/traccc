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
    const default_detector::view& det, const bfield& field,
    const edm::track_candidate_container<default_algebra>::const_view&
        track_candidates) const {

    // Run the track fitting.
    if (field.is<const_bfield_backend_t<scalar>>()) {
        return details::kalman_fitting<default_detector::device>(
            det, field.as<const_bfield_backend_t<scalar>>(), track_candidates,
            m_config, m_mr, m_copy.get(), m_stream, m_warp_size);
    } else if (field.is<cuda::inhom_bfield_backend_t<scalar>>()) {
        return details::kalman_fitting<default_detector::device>(
            det, field.as<cuda::inhom_bfield_backend_t<scalar>>(),
            track_candidates, m_config, m_mr, m_copy.get(), m_stream,
            m_warp_size);
    } else {
        throw std::invalid_argument(
            "Unsupported b-field type received in "
            "traccc::cuda::kalman_fitting_algorithm");
    }
}

}  // namespace traccc::cuda
