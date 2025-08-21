/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/bfield/magnetic_field_types.hpp"
#include "traccc/fitting/details/kalman_fitting.hpp"
#include "traccc/fitting/details/kalman_fitting_types.hpp"
#include "traccc/fitting/kalman_fitting_algorithm.hpp"

// System include(s).
#include <stdexcept>

namespace traccc::host {

kalman_fitting_algorithm::output_type kalman_fitting_algorithm::operator()(
    const telescope_detector::host& det, const magnetic_field& bfield,
    const edm::track_candidate_container<default_algebra>::const_view&
        track_candidates) const {

    // Perform the track fitting using the appropriate templated implementation.
    return magnetic_field_visitor<bfield_type_list<scalar>>(
        bfield, [&]<typename bfield_view_t>(const bfield_view_t& bfield_view) {
            traccc::details::kalman_fitter_t<telescope_detector::host,
                                             bfield_view_t>
                fitter{det, bfield_view, m_config};
            return details::kalman_fitting<default_algebra>(
                fitter, track_candidates, m_mr.get(), m_copy.get());
        });
}

}  // namespace traccc::host
