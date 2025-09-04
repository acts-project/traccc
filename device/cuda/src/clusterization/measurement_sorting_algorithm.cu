/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "../utils/cuda_error_handling.hpp"
#include "../utils/get_size.hpp"
#include "../utils/utils.hpp"
#include "traccc/cuda/clusterization/measurement_sorting_algorithm.hpp"

// Thrust include(s).
#include <thrust/sort.h>

// System include(s).
#include <memory_resource>

namespace traccc::cuda {

measurement_sorting_algorithm::measurement_sorting_algorithm(
    const traccc::memory_resource& mr, vecmem::copy& copy, stream& str,
    std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)), m_mr{mr}, m_copy{copy}, m_stream{str} {}

measurement_sorting_algorithm::output_type
measurement_sorting_algorithm::operator()(
    const measurement_collection_types::view& measurements_view) const {
    // Get a convenience variable for the stream that we'll be using.
    cudaStream_t stream = details::get_stream(m_stream);

    // Staging area for copying sizes from device to host
    vecmem::unique_alloc_ptr<unsigned int> size_staging_ptr =
        vecmem::make_unique_alloc<unsigned int>(*(m_mr.host));

    // Copy the number of measurements to the host
    const measurement_collection_types::view::size_type n_measurements =
        get_size(measurements_view, size_staging_ptr.get(), stream);

    // Sort the measurements in place
    thrust::sort(
        thrust::cuda::par_nosync(std::pmr::polymorphic_allocator(&(m_mr.main)))
            .on(stream),
        measurements_view.ptr(), measurements_view.ptr() + n_measurements,
        measurement_sort_comp());

    // Return the view of the sorted measurements.
    return measurements_view;
}

}  // namespace traccc::cuda
