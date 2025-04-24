/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/fitting/details/fit_tracks.hpp"
#include "traccc/fitting/triplet_fit/triplet_fitter.hpp"
#include "traccc/fitting/triplet_fitting_algorithm.hpp"

// Detray include(s).
// #include <detray/navigation/navigator.hpp>
// #include <detray/propagator/rk_stepper.hpp>

#include <detray/core/detector.hpp>
#include <detray/detectors/bfield.hpp>
#include <traccc/geometry/detector.hpp>

namespace traccc::host {

triplet_fitting_algorithm::output_type triplet_fitting_algorithm::operator()(
    const default_detector::host& det,
    const detray::bfield::const_field_t<
        traccc::default_detector::host::scalar_type>::view_t& field,
    const track_candidate_container_types::const_view& track_candidates) const {

    using scalar_type = traccc::default_detector::host::scalar_type;

    // Create the fitter object.
    triplet_fitter<const traccc::default_detector::host,
                   detray::bfield::const_field_t<scalar_type>::view_t>
        fitter{det, field, m_config};

    // Perform the track fitting using a common, templated function.
    return details::fit_tracks(fitter, track_candidates, m_mr.get(),
                               m_copy.get());
}

}  // namespace traccc::host
