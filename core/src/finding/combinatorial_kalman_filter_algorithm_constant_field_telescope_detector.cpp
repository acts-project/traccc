/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/finding/combinatorial_kalman_filter_algorithm.hpp"
#include "traccc/finding/details/find_tracks.hpp"
#include "traccc/utils/bfield.hpp"
#include "traccc/utils/propagation.hpp"

namespace {
using detector_type = traccc::telescope_detector;
using scalar_type = detector_type::host::scalar_type;
using bfield_type = covfie::field<traccc::const_bfield_backend_t<scalar_type>>;
}  // namespace

namespace traccc::host {
combinatorial_kalman_filter_algorithm::output_type
combinatorial_kalman_filter_algorithm::operator()(
    const detector_type::host& det, const bfield_type::view_t& field,
    const measurement_collection_types::const_view& measurements,
    const bound_track_parameters_collection_types::const_view& seeds) const {

    // Perform the track finding using the templated implementation.
    return details::find_tracks<
        detray::rk_stepper<bfield_type::view_t,
                           detector_type::host::algebra_type,
                           detray::constrained_step<scalar_type>>,
        detray::navigator<const detector_type::host,
                          traccc::detail::ckf_nav_cache_size>>(
        det, field, measurements, seeds, m_config);
}

}  // namespace traccc::host
