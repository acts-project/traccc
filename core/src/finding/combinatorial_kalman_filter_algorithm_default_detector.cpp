/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/bfield/magnetic_field_types.hpp"
#include "traccc/finding/combinatorial_kalman_filter_algorithm.hpp"
#include "traccc/finding/details/combinatorial_kalman_filter.hpp"

// System include(s).
#include <stdexcept>

namespace traccc::host {

combinatorial_kalman_filter_algorithm::output_type
combinatorial_kalman_filter_algorithm::operator()(
    const default_detector::host& det, const magnetic_field& bfield,
    const measurement_collection_types::const_view& measurements,
    const bound_track_parameters_collection_types::const_view& seeds) const {

    // Perform the track finding using the appropriate templated implementation.
    return magnetic_field_visitor<bfield_type_list<scalar>>(
        bfield, [&]<typename bfield_view_t>(const bfield_view_t& bfield_view) {
            return details::combinatorial_kalman_filter(
                det, bfield_view, measurements, seeds, m_config, m_mr.get(),
                logger());
        });
}

}  // namespace traccc::host
