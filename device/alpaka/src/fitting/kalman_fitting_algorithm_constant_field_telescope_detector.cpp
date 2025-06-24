/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../utils/get_queue.hpp"
#include "kalman_fitting.hpp"
#include "traccc/alpaka/fitting/kalman_fitting_algorithm.hpp"

namespace traccc::alpaka {

kalman_fitting_algorithm::output_type kalman_fitting_algorithm::operator()(
    const telescope_detector::view& det,
    const covfie::field<traccc::const_bfield_backend_t<
        telescope_detector::device::scalar_type>>::view_t& field,
    const edm::track_candidate_container<default_algebra>::const_view&
        track_candidates) const {

    // Run the track fitting.
    return details::kalman_fitting<telescope_detector::device>(
        det, field, track_candidates, m_config, m_mr, m_copy.get(),
        details::get_queue(m_queue.get()));
}

}  // namespace traccc::alpaka
