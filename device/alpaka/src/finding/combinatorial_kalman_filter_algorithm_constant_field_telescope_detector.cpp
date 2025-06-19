/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../utils/get_queue.hpp"
#include "find_tracks.hpp"
#include "traccc/alpaka/finding/combinatorial_kalman_filter_algorithm.hpp"

// Project include(s).
#include "traccc/utils/bfield.hpp"
#include "traccc/utils/propagation.hpp"

namespace traccc::alpaka {

combinatorial_kalman_filter_algorithm::output_type
combinatorial_kalman_filter_algorithm::operator()(
    const telescope_detector::view& det,
    const covfie::field<traccc::const_bfield_backend_t<
        telescope_detector::device::scalar_type>>::view_t& field,
    const measurement_collection_types::const_view& measurements,
    const bound_track_parameters_collection_types::const_view& seeds) const {

    using scalar_type = telescope_detector::device::scalar_type;

    // Perform the track finding using the templated implementation.
    return details::find_tracks<
        detray::rk_stepper<
            covfie::field<traccc::const_bfield_backend_t<scalar_type>>::view_t,
            telescope_detector::device::algebra_type,
            detray::constrained_step<scalar_type>>,
        detray::navigator<const telescope_detector::device>>(
        det, field, measurements, seeds, m_config, m_mr, m_copy, logger(),
        details::get_queue(m_queue.get()));
}

}  // namespace traccc::alpaka
