/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/finding/combinatorial_kalman_filter_algorithm.hpp"
#include "traccc/finding/details/combinatorial_kalman_filter.hpp"

// System include(s).
#include <stdexcept>

namespace traccc::host {

combinatorial_kalman_filter_algorithm::output_type
combinatorial_kalman_filter_algorithm::operator()(
    const telescope_detector::host& det, const bfield& field,
    const measurement_collection_types::const_view& measurements,
    const bound_track_parameters_collection_types::const_view& seeds) const {

    // Perform the track finding using the appropriate templated implementation.
    if (field.is<const_bfield_backend_t<scalar>>()) {
        return details::combinatorial_kalman_filter(
            det, field.as<const_bfield_backend_t<scalar>>(), measurements,
            seeds, m_config, m_mr.get(), logger());
    } else if (field.is<inhom_bfield_backend_t<scalar>>()) {
        return details::combinatorial_kalman_filter(
            det, field.as<inhom_bfield_backend_t<scalar>>(), measurements,
            seeds, m_config, m_mr.get(), logger());
    } else {
        throw std::invalid_argument(
            "Unsupported b-field type received in "
            "traccc::host::combinatorial_kalman_filter_algorithm");
    }
}

}  // namespace traccc::host
