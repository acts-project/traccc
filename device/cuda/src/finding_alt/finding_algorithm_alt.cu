/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/cuda/finding_alt/finding_algorithm_alt.hpp"
#include "traccc/cuda/utils/definitions.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/finding/device/make_module_map.hpp"
#include "traccc/finding_alt/device/build_tracks.hpp"
#include "traccc/finding_alt/device/find_tracks.hpp"

// detray include(s).
#include "detray/core/detector.hpp"
#include "detray/detectors/detector_metadata.hpp"
#include "detray/masks/unbounded.hpp"
#include "detray/propagator/navigator.hpp"
#include "detray/propagator/rk_stepper.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_buffer.hpp>
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/jagged_device_vector.hpp>
#include <vecmem/containers/vector.hpp>

// System include(s).
#include <vector>

namespace traccc::cuda {

namespace kernels {

/// CUDA kernel for running @c traccc::device::make_module_map
__global__ void make_module_map_alt(
    measurement_container_types::const_view measurements_view,
    vecmem::data::vector_view<thrust::pair<geometry_id, unsigned int>>
        module_map_view) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    device::make_module_map(gid, measurements_view, module_map_view);
}

/// CUDA kernel for running @c traccc::device::find_tracks
template <typename propagator_t, typename config_t>
__global__ void find_tracks(
    const config_t cfg,
    typename propagator_t::detector_type::detector_view_type det_data,
    measurement_container_types::const_view measurements_view,
    vecmem::data::vector_view<const thrust::pair<geometry_id, unsigned int>>
        module_map_view,
    bound_track_parameters_collection_types::const_view seeds_view,
    vecmem::data::vector_view<const candidate_link_alt> links_view,
    vecmem::data::vector_view<
        const typename candidate_link_alt::link_index_type>
        tips_view) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    device::find_tracks<propagator_t, config_t>(
        gid, cfg, det_data, measurements_view, module_map_view, seeds_view,
        links_view, tips_view);
}

/// CUDA kernel for running @c traccc::device::build_tracks
__global__ void build_tracks(
    measurement_container_types::const_view measurements_view,
    bound_track_parameters_collection_types::const_view seeds_view,
    vecmem::data::vector_view<const candidate_link_alt> links_view,
    vecmem::data::vector_view<
        const typename candidate_link_alt::link_index_type>
        tips_view,
    track_candidate_container_types::view track_candidates_view) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    device::build_tracks(gid, measurements_view, seeds_view, links_view,
                         tips_view, track_candidates_view);
}

}  // namespace kernels

template <typename stepper_t, typename navigator_t>
finding_algorithm_alt<stepper_t, navigator_t>::finding_algorithm_alt(
    const config_type& cfg, const traccc::memory_resource& mr)
    : m_cfg(cfg), m_mr(mr) {

    // Initialize m_copy ptr based on memory resources that were given
    if (mr.host) {
        m_copy = std::make_unique<vecmem::cuda::copy>();
    } else {
        m_copy = std::make_unique<vecmem::copy>();
    }
};

template <typename stepper_t, typename navigator_t>
track_candidate_container_types::buffer
finding_algorithm_alt<stepper_t, navigator_t>::operator()(
    const typename detector_type::detector_view_type& det_view,
    const vecmem::data::jagged_vector_view<
        typename navigator_t::intersection_type>& navigation_buffer,
    const typename measurement_container_types::const_view& measurements,
    bound_track_parameters_collection_types::buffer&& seeds_buffer) const {

    // Copy setup
    m_copy->setup(seeds_buffer);
    m_copy->setup(navigation_buffer);

    /*****************************************************************
     * Kernel1: Create module map
     *****************************************************************/

    vecmem::data::vector_buffer<thrust::pair<geometry_id, unsigned int>>
        module_map_buffer{measurements.headers.size(), m_mr.main};

    unsigned int nThreads = WARP_SIZE * 2;
    unsigned int nBlocks = (module_map_buffer.size() + nThreads - 1) / nThreads;
    kernels::make_module_map_alt<<<nBlocks, nThreads>>>(measurements,
                                                        module_map_buffer);
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    vecmem::device_vector<thrust::pair<geometry_id, unsigned int>> module_map(
        module_map_buffer);
    thrust::sort(thrust::device, module_map.begin(), module_map.end());

    /*****************************************************************
     * Kernel2: Find tracks
     *****************************************************************/

    vecmem::data::vector_buffer<const candidate_link_alt> links_buffer(
        10000, m_mr.main, vecmem::data::buffer_type::resizable);
    vecmem::data::vector_buffer<const candidate_link_alt::link_index_type>
        tips_buffer(1000, m_mr.main, vecmem::data::buffer_type::resizable);

    const unsigned int n_seeds = m_copy->get_size(seeds_buffer);
    if (n_seeds > 0) {
        nThreads = WARP_SIZE * 2;
        nBlocks = (n_seeds + nThreads - 1) / nThreads;

        kernels::find_tracks<propagator_type, config_type>
            <<<nBlocks, nThreads>>>(m_cfg, det_view, measurements,
                                    module_map_buffer, seeds_buffer,
                                    links_buffer, tips_buffer);

        CUDA_ERROR_CHECK(cudaGetLastError());
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    }

    /*****************************************************************
     * Kernel3: Build tracks
     *****************************************************************/

    const unsigned int n_tips = m_copy->get_size(tips_buffer);

    // Create track candidate buffer
    track_candidate_container_types::buffer track_candidates_buffer{
        {n_tips, m_mr.main},
        {std::vector<std::size_t>(n_tips, m_cfg.max_track_candidates_per_track),
         m_mr.main, m_mr.host, vecmem::data::buffer_type::resizable}};

    m_copy->setup(track_candidates_buffer.headers);
    m_copy->setup(track_candidates_buffer.items);

    // @Note: nBlocks can be zero in case there is no tip. This happens when
    // chi2_max config is set tightly and no tips are found
    if (n_tips > 0) {
        nThreads = WARP_SIZE * 2;
        nBlocks = (n_tips + nThreads - 1) / nThreads;
        kernels::build_tracks<<<nBlocks, nThreads>>>(measurements, seeds_buffer,
                                                     links_buffer, tips_buffer,
                                                     track_candidates_buffer);

        CUDA_ERROR_CHECK(cudaGetLastError());
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    }

    return track_candidates_buffer;
}

// Explicit template instantiation
using device_detector_type =
    detray::detector<detray::detector_registry::template telescope_detector<
                         detray::rectangle2D<>>,
                     covfie::field_view, detray::device_container_types>;
using rk_stepper_type = detray::rk_stepper<
    covfie::field<device_detector_type::bfield_backend_type>::view_t,
    transform3, detray::constrained_step<>>;
using device_navigator_type = detray::navigator<const device_detector_type>;
template class finding_algorithm_alt<rk_stepper_type, device_navigator_type>;

}  // namespace traccc::cuda
