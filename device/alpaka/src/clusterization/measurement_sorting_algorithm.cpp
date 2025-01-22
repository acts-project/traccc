/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/alpaka/clusterization/measurement_sorting_algorithm.hpp"

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <thrust/sort.h>
#endif

namespace traccc::alpaka {

measurement_sorting_algorithm::measurement_sorting_algorithm(
    vecmem::copy& copy, std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)), m_copy{copy} {}

measurement_sorting_algorithm::output_type
measurement_sorting_algorithm::operator()(
    const measurement_collection_types::view& measurements_view) const {

    // Get the number of measurements. This is necessary because the input
    // container may not be fixed sized. And we can't give invalid pointers /
    // iterators to Thrust.
    const measurement_collection_types::view::size_type n_measurements =
        m_copy.get().get_size(measurements_view);

    // Sort the measurements in place
    // TODO: This obviously won't work for HIP, but we don't have a HIP
    // equivalent of Thrust yet.
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    thrust::sort(thrust::device, measurements_view.ptr(),
                 measurements_view.ptr() + n_measurements,
                 measurement_sort_comp());
#elif !defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    std::sort(measurements_view.ptr(), measurements_view.ptr() + n_measurements,
              measurement_sort_comp());
#endif

    // Return the view of the sorted measurements.
    return measurements_view;
}

}  // namespace traccc::alpaka
