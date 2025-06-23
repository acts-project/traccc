/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/finding/combinatorial_kalman_filter_algorithm.hpp"
#include "traccc/finding/details/combinatorial_kalman_filter.hpp"

namespace traccc::host {

combinatorial_kalman_filter_algorithm::output_type
combinatorial_kalman_filter_algorithm::operator()(
    const default_detector::host& det,
    const covfie::field<traccc::const_bfield_backend_t<
        default_detector::host::scalar_type>>::view_t& field,
    const measurement_collection_types::const_view& measurements,
    const bound_track_parameters_collection_types::const_view& seeds) const {

    // Perform the track finding using the templated implementation.
    return details::combinatorial_kalman_filter(det, field, measurements, seeds,
                                                m_config, m_mr.get(), logger());
}

}  // namespace traccc::host
