/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "fit_tracks.hpp"
#include "traccc/fitting/kalman_filter/kalman_fitter.hpp"
#include "traccc/fitting/kalman_fitting_algorithm.hpp"

// Detray include(s).
#include <detray/navigation/navigator.hpp>
#include <detray/propagator/rk_stepper.hpp>

namespace traccc::host {

kalman_fitting_algorithm::output_type kalman_fitting_algorithm::operator()(
    const telescope_detector::host& det,
    const detray::bfield::const_field_t::view_t& field,
    const track_candidate_container_types::const_view& track_candidates) const {

    // Set up the fitter type(s).
    using stepper_type =
        detray::rk_stepper<detray::bfield::const_field_t::view_t,
                           traccc::telescope_detector::host::algebra_type,
                           detray::constrained_step<>>;
    using navigator_type =
        detray::navigator<const traccc::telescope_detector::host>;
    using fitter_type = kalman_fitter<stepper_type, navigator_type>;

    // Perform the track fitting using a common, templated function.
    return details::fit_tracks<fitter_type>(det, field, track_candidates,
                                            m_config);
}

}  // namespace traccc::host
