/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../utils/bfield.cuh"
#include "combinatorial_kalman_filter.cuh"
#include "traccc/cuda/finding/combinatorial_kalman_filter_algorithm.hpp"
#include "traccc/finding/device/tags.hpp"

// System include(s).
#include <stdexcept>

namespace traccc::cuda {

combinatorial_kalman_filter_algorithm::unfitted_output_type
combinatorial_kalman_filter_algorithm::operator()(
    const default_detector::view& det, const bfield& field,
    const measurement_collection_types::const_view& measurements,
    const bound_track_parameters_collection_types::const_view& seeds,
    device::finding_return_unfitted&&) const {

    // Perform the track finding using the templated implementation.
    if (field.is<const_bfield_backend_t<scalar>>()) {
        return details::combinatorial_kalman_filter<default_detector::device>(
            det, field.as<const_bfield_backend_t<scalar>>(), measurements,
            seeds, m_config, m_mr, m_copy, logger(), m_stream, m_warp_size,
            device::finding_return_unfitted{});
    } else if (field.is<cuda::inhom_bfield_backend_t<scalar>>()) {
        return details::combinatorial_kalman_filter<default_detector::device>(
            det, field.as<cuda::inhom_bfield_backend_t<scalar>>(), measurements,
            seeds, m_config, m_mr, m_copy, logger(), m_stream, m_warp_size,
            device::finding_return_unfitted{});
    } else {
        throw std::invalid_argument(
            "Unsupported b-field type received in "
            "traccc::cuda::combinatorial_kalman_filter_algorithm");
    }
}

combinatorial_kalman_filter_algorithm::fitted_output_type
combinatorial_kalman_filter_algorithm::operator()(
    const default_detector::view& det, const bfield& field,
    const measurement_collection_types::const_view& measurements,
    const bound_track_parameters_collection_types::const_view& seeds,
    device::finding_return_fitted&&) const {

    // Perform the track finding using the templated implementation.
    if (field.is<const_bfield_backend_t<scalar>>()) {
        return details::combinatorial_kalman_filter<default_detector::device>(
            det, field.as<const_bfield_backend_t<scalar>>(), measurements,
            seeds, m_config, m_mr, m_copy, logger(), m_stream, m_warp_size,
            device::finding_return_fitted{});
    } else if (field.is<cuda::inhom_bfield_backend_t<scalar>>()) {
        return details::combinatorial_kalman_filter<default_detector::device>(
            det, field.as<cuda::inhom_bfield_backend_t<scalar>>(), measurements,
            seeds, m_config, m_mr, m_copy, logger(), m_stream, m_warp_size,
            device::finding_return_fitted{});
    } else {
        throw std::invalid_argument(
            "Unsupported b-field type received in "
            "traccc::cuda::combinatorial_kalman_filter_algorithm");
    }
}

}  // namespace traccc::cuda
