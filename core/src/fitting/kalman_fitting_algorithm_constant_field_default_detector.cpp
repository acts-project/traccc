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
#include "traccc/utils/bfield.hpp"
#include "traccc/utils/propagation.hpp"

namespace {
using detector_type = traccc::default_detector;
using scalar_type = detector_type::host::scalar_type;
using bfield_type = covfie::field<traccc::const_bfield_backend_t<scalar_type>>;
}  // namespace

namespace traccc::host {

kalman_fitting_algorithm::output_type kalman_fitting_algorithm::operator()(
    const detector_type::host& det, const bfield_type::view_t& field,
    const track_candidate_container_types::const_view& track_candidates) const {

    // Create the fitter object.
    kalman_fitter<detray::rk_stepper<bfield_type::view_t,
                                     detector_type::host::algebra_type,
                                     detray::constrained_step<scalar_type>>,
                  detray::navigator<const detector_type::host>>
        fitter{det, field, m_config};

    // Perform the track fitting using a common, templated function.
    return details::fit_tracks(fitter, track_candidates, m_mr.get(),
                               m_copy.get());
}

}  // namespace traccc::host
