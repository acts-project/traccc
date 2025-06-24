/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/fitting/details/kalman_fitting.hpp"
#include "traccc/fitting/details/kalman_fitting_types.hpp"
#include "traccc/fitting/kalman_fitting_algorithm.hpp"

namespace traccc::host {

kalman_fitting_algorithm::output_type kalman_fitting_algorithm::operator()(
    const default_detector::host& det,
    const covfie::field<traccc::const_bfield_backend_t<
        default_detector::host::scalar_type>>::view_t& field,
    const edm::track_candidate_container<default_algebra>::const_view&
        track_candidates) const {

    // Create the fitter object.
    traccc::details::kalman_fitter_t<
        default_detector::host,
        covfie::field<traccc::const_bfield_backend_t<
            default_detector::host::scalar_type>>::view_t>
        fitter{det, field, m_config};

    // Perform the track fitting using a common, templated function.
    return details::kalman_fitting(fitter, track_candidates, m_mr.get(),
                                   m_copy.get());
}

}  // namespace traccc::host
