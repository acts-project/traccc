/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "kalman_fitting.cuh"
#include "traccc/cuda/fitting/kalman_fitting_algorithm.hpp"

// Project include(s).
#include "traccc/fitting/kalman_filter/kalman_fitter.hpp"
#include "traccc/utils/propagation.hpp"

namespace traccc::cuda {

kalman_fitting_algorithm::output_type kalman_fitting_algorithm::operator()(
    const default_detector::view& det,
    const covfie::field<traccc::const_bfield_backend_t<
        default_detector::device::scalar_type>>::view_t& field,
    const edm::track_candidate_container<default_algebra>::const_view&
        track_candidates) const {

    using scalar_type = default_detector::device::scalar_type;

    using bfield_type =
        covfie::field<traccc::const_bfield_backend_t<scalar_type>>;

    // Construct the fitter type.
    using stepper_type =
        detray::rk_stepper<bfield_type::view_t,
                           default_detector::device::algebra_type,
                           detray::constrained_step<scalar_type>>;
    using navigator_type = detray::navigator<const default_detector::device>;
    using fitter_type = kalman_fitter<stepper_type, navigator_type>;

    // Run the track fitting.
    return details::kalman_fitting<fitter_type>(det, field, track_candidates,
                                                m_config, m_mr, m_copy.get(),
                                                m_stream, m_warp_size);
}

}  // namespace traccc::cuda
