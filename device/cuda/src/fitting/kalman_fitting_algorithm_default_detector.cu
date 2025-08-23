/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../utils/magnetic_field_types.hpp"
#include "kalman_fitting.cuh"
#include "traccc/cuda/fitting/kalman_fitting_algorithm.hpp"

// Project include(s).
#include "traccc/bfield/magnetic_field_types.hpp"

namespace traccc::cuda {

kalman_fitting_algorithm::output_type kalman_fitting_algorithm::operator()(
    const default_detector::view& det, const magnetic_field& bfield,
    const edm::track_candidate_container<default_algebra>::const_view&
        track_candidates) const {

    // Run the track fitting.
    return magnetic_field_visitor<cuda::bfield_type_list<scalar>>(
        bfield, [&]<typename bfield_view_t>(const bfield_view_t& bfield_view) {
            return details::kalman_fitting<default_detector::device>(
                det, bfield_view, track_candidates, m_config, m_mr,
                m_copy.get(), m_stream, m_warp_size);
        });
}

}  // namespace traccc::cuda
