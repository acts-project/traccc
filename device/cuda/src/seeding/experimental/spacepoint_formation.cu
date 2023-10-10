/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../../utils/utils.hpp"
#include "traccc/cuda/seeding/experimental/spacepoint_formation.hpp"
#include "traccc/cuda/utils/barrier.hpp"
#include "traccc/cuda/utils/definitions.hpp"
#include "traccc/seeding/device/experimental/form_spacepoints.hpp"

// detray include(s).
#include "detray/core/detector.hpp"
#include "detray/detectors/telescope_metadata.hpp"

namespace traccc::cuda::experimental {

namespace kernels {

template <typename detector_t>
__global__ void form_spacepoints(
    typename detector_t::detector_view_type<detray::bfield::const_bknd_t>
        det_data,
    measurement_collection_types::const_view measurements_view,
    spacepoint_collection_types::view spacepoints_view) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    device::experimental::form_spacepoints<detector_t>(
        gid, det_data, measurements_view, spacepoints_view);
}

}  // namespace kernels

template <typename detector_t>
spacepoint_formation<detector_t>::spacepoint_formation(
    const traccc::memory_resource& mr, vecmem::copy& copy, stream& str)
    : m_mr(mr), m_copy(copy), m_stream(str) {}

template <typename detector_t>
spacepoint_collection_types::buffer
spacepoint_formation<detector_t>::operator()(
    const typename detector_t::detector_view_type<detray::bfield::const_bknd_t>&
        det_view,
    const measurement_collection_types::const_view& measurements_view) const {

    // Get a convenience variable for the stream that we'll be using.
    cudaStream_t stream = details::get_stream(m_stream);

    const std::size_t n_measurements = m_copy.get_size(measurements_view);

    spacepoint_collection_types::buffer spacepoints_buffer(
        n_measurements, m_mr.main, vecmem::data::buffer_type::resizable);
    m_copy.setup(spacepoints_buffer);

    unsigned int nThreads = WARP_SIZE * 2;
    unsigned int nBlocks = (n_measurements + nThreads - 1) / nThreads;

    kernels::form_spacepoints<detector_t><<<nBlocks, nThreads, 0, stream>>>(
        det_view, measurements_view, spacepoints_buffer);

    CUDA_ERROR_CHECK(cudaGetLastError());

    return spacepoints_buffer;
}

using telescope_detector_type =
    detray::detector<detray::telescope_metadata<detray::rectangle2D<>>,
                     covfie::field_view<detray::bfield::const_bknd_t>,
                     detray::device_container_types>;
template class spacepoint_formation<telescope_detector_type>;

}  // namespace traccc::cuda::experimental