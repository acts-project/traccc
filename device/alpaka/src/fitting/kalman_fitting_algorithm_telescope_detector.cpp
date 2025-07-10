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

// Project include(s).
#include "traccc/bfield/magnetic_field_types.hpp"

namespace traccc::alpaka {

kalman_fitting_algorithm::output_type kalman_fitting_algorithm::operator()(
    const telescope_detector::view& det, const magnetic_field& bfield,
    const edm::track_candidate_container<default_algebra>::const_view&
        track_candidates) const {

    // Run the track fitting.
    if (bfield.is<const_bfield_backend_t<scalar>>()) {
        return details::kalman_fitting<telescope_detector::device>(
            det, bfield.as_view<const_bfield_backend_t<scalar>>(),
            track_candidates, m_config, m_mr, m_copy.get(),
            details::get_queue(m_queue.get()));
    } else {
        throw std::invalid_argument(
            "Unsupported b-field type received in "
            "traccc::alpaka::kalman_fitting_algorithm");
    }
}

}  // namespace traccc::alpaka
