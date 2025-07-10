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
    const telescope_detector::host& det, const magnetic_field& bfield,
    const measurement_collection_types::const_view& measurements,
    const bound_track_parameters_collection_types::const_view& seeds) const {

    // Perform the track finding using the appropriate templated implementation.
    if (bfield.is<const_bfield_backend_t<scalar>>()) {
        return details::combinatorial_kalman_filter(
            det, bfield.as_view<const_bfield_backend_t<scalar>>(), measurements,
            seeds, m_config, m_mr.get(), logger());
    } else if (bfield.is<host::inhom_bfield_backend_t<scalar>>()) {
        return details::combinatorial_kalman_filter(
            det, bfield.as_view<host::inhom_bfield_backend_t<scalar>>(),
            measurements, seeds, m_config, m_mr.get(), logger());
    } else {
        throw std::invalid_argument(
            "Unsupported b-field type received in "
            "traccc::host::combinatorial_kalman_filter_algorithm");
    }
}

}  // namespace traccc::host
