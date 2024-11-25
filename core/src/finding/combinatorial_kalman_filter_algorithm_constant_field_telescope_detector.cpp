/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/finding/combinatorial_kalman_filter_algorithm.hpp"
#include "traccc/finding/details/find_tracks.hpp"

// Detray include(s).
#include <detray/detectors/bfield.hpp>
#include <detray/navigation/navigator.hpp>
#include <detray/propagator/propagator.hpp>
#include <detray/propagator/rk_stepper.hpp>

namespace traccc::host {

combinatorial_kalman_filter_algorithm::output_type
combinatorial_kalman_filter_algorithm::operator()(
    const telescope_detector::host& det,
    const detray::bfield::const_field_t::view_t& field,
    const measurement_collection_types::const_view& measurements,
    const bound_track_parameters_collection_types::const_view& seeds) const {

    // Perform the track finding using the templated implementation.
    return details::find_tracks<
        detray::rk_stepper<detray::bfield::const_field_t::view_t,
                           telescope_detector::host::algebra_type,
                           detray::constrained_step<>>,
        detray::navigator<const telescope_detector::host>>(
        det, field, measurements, seeds, m_config);
}

}  // namespace traccc::host
