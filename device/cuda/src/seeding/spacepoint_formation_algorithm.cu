/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../utils/cuda_error_handling.hpp"
#include "../utils/get_size.hpp"
#include "../utils/global_index.hpp"
#include "../utils/utils.hpp"
#include "traccc/cuda/seeding/spacepoint_formation_algorithm.hpp"

// Project include(s).
#include "traccc/geometry/detector.hpp"
#include "traccc/seeding/device/form_spacepoints.hpp"

namespace traccc::cuda {
namespace kernels {

template <typename detector_t>
__global__ void __launch_bounds__(1024, 1)
    form_spacepoints(typename detector_t::view det_view,
                     measurement_collection_types::const_view measurements_view,
                     edm::spacepoint_collection::view spacepoints_view)
    requires(traccc::is_detector_traits<detector_t>)
{

    device::form_spacepoints<detector_t>(details::global_index1(), det_view,
                                         measurements_view, spacepoints_view);
}

}  // namespace kernels

spacepoint_formation_algorithm::spacepoint_formation_algorithm(
    const traccc::memory_resource& mr, vecmem::copy& copy, stream& str,
    std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)), m_mr(mr), m_copy(copy), m_stream(str) {}

edm::spacepoint_collection::buffer spacepoint_formation_algorithm::operator()(
    const detector_buffer& detector,
    const measurement_collection_types::const_view& measurements_view) const {

    // Get a convenience variable for the stream that we'll be using.
    cudaStream_t stream = details::get_stream(m_stream);

    // Staging area for copying sizes from device to host
    vecmem::unique_alloc_ptr<unsigned int> size_staging_ptr =
        vecmem::make_unique_alloc<unsigned int>(*(m_mr.host));

    // Get the number of measurements.
    const measurement_collection_types::const_view::size_type num_measurements =
        get_size(measurements_view, size_staging_ptr.get(), stream);

    // Create the result buffer.
    edm::spacepoint_collection::buffer spacepoints(
        num_measurements, m_mr.main, vecmem::data::buffer_type::resizable);
    m_copy.get().setup(spacepoints)->ignore();

    // If there are no measurements, we can conclude here.
    if (num_measurements == 0) {
        return spacepoints;
    }

    // Launch parameters for the kernel.
    const unsigned int blockSize = 256;
    const unsigned int nBlocks = (num_measurements + blockSize - 1) / blockSize;

    // Launch the spacepoint formation kernel.
    detector_buffer_visitor<detector_type_list>(
        detector, [&]<typename detector_traits_t>(
                      const typename detector_traits_t::view& det) {
            kernels::form_spacepoints<detector_traits_t>
                <<<nBlocks, blockSize, 0, stream>>>(det, measurements_view,
                                                    spacepoints);
        });

    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

    // Return the reconstructed spacepoints.
    return spacepoints;
}
}  // namespace traccc::cuda
