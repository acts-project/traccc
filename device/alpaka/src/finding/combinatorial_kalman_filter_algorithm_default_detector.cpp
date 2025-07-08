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
    const default_detector::view& det, const bfield& field,
    const measurement_collection_types::const_view& measurements,
    const bound_track_parameters_collection_types::const_view& seeds) const {

    // Perform the track finding using the templated implementation.
    if (field.is<const_bfield_backend_t<scalar>>()) {
        return details::combinatorial_kalman_filter<default_detector::device>(
            det, field.as<const_bfield_backend_t<scalar>>(), measurements,
            seeds, m_config, m_mr, m_copy, logger(),
            details::get_queue(m_queue.get()));
    } else {
        throw std::invalid_argument(
            "Unsupported b-field type received in "
            "traccc::alpaka::combinatorial_kalman_filter_algorithm");
    }
}

}  // namespace traccc::alpaka
