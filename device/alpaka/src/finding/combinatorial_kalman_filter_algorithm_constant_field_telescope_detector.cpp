/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../utils/get_queue.hpp"
#include "combinatorial_kalman_filter.hpp"
#include "traccc/alpaka/finding/combinatorial_kalman_filter_algorithm.hpp"

namespace traccc::alpaka {

combinatorial_kalman_filter_algorithm::output_type
combinatorial_kalman_filter_algorithm::operator()(
    const telescope_detector::view& det,
    const covfie::field<traccc::const_bfield_backend_t<
        telescope_detector::device::scalar_type>>::view_t& field,
    const measurement_collection_types::const_view& measurements,
    const bound_track_parameters_collection_types::const_view& seeds) const {

    // Perform the track finding using the templated implementation.
    return details::combinatorial_kalman_filter<telescope_detector::device>(
        det, field, measurements, seeds, m_config, m_mr, m_copy, logger(),
        details::get_queue(m_queue.get()));
}

}  // namespace traccc::alpaka
