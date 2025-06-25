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

// System include(s).
#include <stdexcept>

namespace traccc::host {

kalman_fitting_algorithm::output_type kalman_fitting_algorithm::operator()(
    const telescope_detector::host& det, const bfield& field,
    const edm::track_candidate_container<default_algebra>::const_view&
        track_candidates) const {

    // Perform the track finding using the appropriate templated implementation.
    if (field.is<const_bfield_backend_t<scalar>>()) {
        traccc::details::kalman_fitter_t<
            telescope_detector::host,
            covfie::field<const_bfield_backend_t<scalar>>::view_t>
            fitter{det, field.as<const_bfield_backend_t<scalar>>(), m_config};
        return details::kalman_fitting(fitter, track_candidates, m_mr.get(),
                                       m_copy.get());
    } else if (field.is<inhom_bfield_backend_t<scalar>>()) {
        traccc::details::kalman_fitter_t<
            telescope_detector::host,
            covfie::field<inhom_bfield_backend_t<scalar>>::view_t>
            fitter{det, field.as<inhom_bfield_backend_t<scalar>>(), m_config};
        return details::kalman_fitting(fitter, track_candidates, m_mr.get(),
                                       m_copy.get());
    } else {
        throw std::invalid_argument(
            "Unsupported b-field type received in "
            "traccc::host::kalman_fitting_algorithm");
    }
}

}  // namespace traccc::host
