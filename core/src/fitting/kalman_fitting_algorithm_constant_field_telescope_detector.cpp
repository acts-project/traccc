/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/fitting/details/fit_tracks.hpp"
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

    // Create the fitter object.
    kalman_fitter<
        detray::rk_stepper<detray::bfield::const_field_t::view_t,
                           traccc::telescope_detector::host::algebra_type,
                           detray::constrained_step<>>,
        detray::navigator<const traccc::telescope_detector::host>>
        fitter{det, field, m_config};

    // Perform the track fitting using a common, templated function.
    return details::fit_tracks(fitter, track_candidates, m_mr.get());
}

}  // namespace traccc::host
