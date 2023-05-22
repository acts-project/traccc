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

// VecMem include(s).
#include <vecmem/containers/data/vector_buffer.hpp>
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/jagged_device_vector.hpp>
#include <vecmem/containers/vector.hpp>

// System include(s).
#include <vector>

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

    // Prepare input parameters with seeds
    bound_track_parameters_collection_types::buffer in_params_buffer(
        m_copy->get_size(seeds_buffer), m_mr.main);
    bound_track_parameters_collection_types::device in_params(in_params_buffer);
    bound_track_parameters_collection_types::device seeds(seeds_buffer);
    thrust::copy(thrust::device, seeds.begin(), seeds.end(), in_params.begin());

    /*****************************************************************
     * Kernel1: Create module map
     *****************************************************************/

    vecmem::data::vector_buffer<thrust::pair<geometry_id, unsigned int>>
        module_map_buffer{measurements.headers.size(), m_mr.main};

    unsigned int nThreads = WARP_SIZE * 2;
    unsigned int nBlocks = (module_map_buffer.size() + nThreads - 1) / nThreads;
    kernels::make_module_map<<<nBlocks, nThreads>>>(measurements,
                                                    module_map_buffer);
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    vecmem::device_vector<thrust::pair<geometry_id, unsigned int>> module_map(
        module_map_buffer);
    thrust::sort(thrust::device, module_map.begin(), module_map.end());

    /*****************************************************************
     * Kernel2: Find tracks
     *****************************************************************/

    /*****************************************************************
     * Kernel3: Build tracks
     *****************************************************************/

    // Create track candidate buffer
    track_candidate_container_types::buffer track_candidates_buffer{
        {n_tips_total, m_mr.main},
        {std::vector<std::size_t>(n_tips_total,
                                  m_cfg.max_num_branches_per_surface),
         m_mr.main, m_mr.host, vecmem::data::buffer_type::resizable}};

    m_copy->setup(track_candidates_buffer.headers);
    m_copy->setup(track_candidates_buffer.items);

    // @Note: nBlocks can be zero in case there is no tip. This happens when
    // chi2_max config is set tightly and no tips are found
    if (n_tips_total > 0) {
        nThreads = WARP_SIZE * 2;
        nBlocks = (n_tips_total + nThreads - 1) / nThreads;
        kernels::build_tracks<<<nBlocks, nThreads>>>(
            measurements, seeds_buffer, links_buffer, param_to_link_buffer,
            tips_buffer, track_candidates_buffer);

        CUDA_ERROR_CHECK(cudaGetLastError());
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    }

    return track_candidates_buffer;
}