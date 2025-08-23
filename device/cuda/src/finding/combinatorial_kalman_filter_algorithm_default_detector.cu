/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../utils/magnetic_field_types.hpp"
#include "combinatorial_kalman_filter.cuh"
#include "traccc/cuda/finding/combinatorial_kalman_filter_algorithm.hpp"

// Project include(s).
#include "traccc/bfield/magnetic_field_types.hpp"

// System include(s).
#include <stdexcept>

namespace traccc::cuda {

combinatorial_kalman_filter_algorithm::output_type
combinatorial_kalman_filter_algorithm::operator()(
    const default_detector::view& det, const magnetic_field& bfield,
    const measurement_collection_types::const_view& measurements,
    const bound_track_parameters_collection_types::const_view& seeds) const {

    // Perform the track finding using the appropriate templated implementation.
    return magnetic_field_visitor<cuda::bfield_type_list<scalar>>(
        bfield, [&]<typename bfield_view_t>(const bfield_view_t& bfield_view) {
            return details::combinatorial_kalman_filter<
                default_detector::device>(det, bfield_view, measurements, seeds,
                                          m_config, m_mr, m_copy, logger(),
                                          m_stream, m_warp_size);
        });
}

}  // namespace traccc::cuda
