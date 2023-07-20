/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/cuda/finding/finding_algorithm.hpp"
#include "traccc/cuda/utils/definitions.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/edm/device/finding_global_counter.hpp"
#include "traccc/finding/candidate_link.hpp"
#include "traccc/finding/device/apply_interaction.hpp"
#include "traccc/finding/device/build_tracks.hpp"
#include "traccc/finding/device/count_measurements.hpp"
#include "traccc/finding/device/count_threads.hpp"
#include "traccc/finding/device/find_tracks.hpp"
#include "traccc/finding/device/make_module_map.hpp"
#include "traccc/finding/device/propagate_to_next_surface.hpp"

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

// Thrust include(s).
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

// System include(s).
#include <vector>

namespace traccc::cuda {

namespace kernels {

/// CUDA kernel for running @c traccc::device::make_module_map
__global__ void make_module_map(
    measurement_container_types::const_view measurements_view,
    vecmem::data::vector_view<thrust::pair<geometry_id, unsigned int>>
        module_map_view) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    device::make_module_map(gid, measurements_view, module_map_view);
}

/// CUDA kernel for running @c traccc::device::apply_interaction
template <typename detector_t>
__global__ void apply_interaction(
    typename detector_t::detector_view_type det_data,
    vecmem::data::jagged_vector_view<detray::intersection2D<
        typename detector_t::surface_type, typename detector_t::transform3>>
        nav_candidates_buffer,
    const int n_params,
    bound_track_parameters_collection_types::view params_view) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    device::apply_interaction<detector_t>(gid, det_data, nav_candidates_buffer,
                                          n_params, params_view);
}

/// CUDA kernel for running @c traccc::device::count_measurements
template <typename detector_t>
__global__ void count_measurements(
    typename detector_t::detector_view_type det_data,
    measurement_container_types::const_view measurements_view,
    vecmem::data::vector_view<const thrust::pair<geometry_id, unsigned int>>
        module_map_view,
    const int n_params,
    bound_track_parameters_collection_types::const_view params_view,
    vecmem::data::vector_view<unsigned int> n_measurements_view,
    unsigned int& n_total_measurements) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    device::count_measurements<detector_t>(
        gid, det_data, measurements_view, module_map_view, n_params,
        params_view, n_measurements_view, n_total_measurements);
}

/// CUDA kernel for running @c traccc::device::count_threads
template <typename config_t>
__global__ void count_threads(
    const config_t cfg,
    vecmem::data::vector_view<const unsigned int> n_measurements_view,
    const unsigned int& n_total_measurements,
    vecmem::data::vector_view<unsigned int> n_threads_view,
    unsigned int& n_measurements_per_thread, unsigned int& n_total_threads) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    device::count_threads<config_t>(gid, cfg, n_measurements_view,
                                    n_total_measurements, n_threads_view,
                                    n_measurements_per_thread, n_total_threads);
}

/// CUDA kernel for running @c traccc::device::find_tracks
template <typename detector_t, typename config_t>
__global__ void find_tracks(
    const config_t cfg, typename detector_t::detector_view_type det_data,
    measurement_container_types::const_view measurements_view,
    vecmem::data::vector_view<const thrust::pair<geometry_id, unsigned int>>
        module_map_view,
    bound_track_parameters_collection_types::const_view in_params_view,
    vecmem::data::vector_view<const unsigned int> n_threads_view,
    const unsigned int step, const unsigned int& n_measurements_per_thread,
    const unsigned int& n_total_threads,
    bound_track_parameters_collection_types::view out_params_view,
    vecmem::data::vector_view<candidate_link> links_view,
    unsigned int& n_candidates) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    device::find_tracks<detector_t, config_t>(
        gid, cfg, det_data, measurements_view, module_map_view, in_params_view,
        n_threads_view, step, n_measurements_per_thread, n_total_threads,
        out_params_view, links_view, n_candidates);
}

/// CUDA kernel for running @c traccc::device::propagate_to_next_surface
template <typename propagator_t, typename config_t>
__global__ void propagate_to_next_surface(
    const config_t cfg,
    typename propagator_t::detector_type::detector_view_type det_data,
    vecmem::data::jagged_vector_view<typename propagator_t::intersection_type>
        nav_candidates_buffer,
    bound_track_parameters_collection_types::const_view in_params_view,
    vecmem::data::vector_view<const candidate_link> links_view,
    const unsigned int step, const unsigned int& n_candidates,
    bound_track_parameters_collection_types::view out_params_view,
    vecmem::data::vector_view<unsigned int> param_to_link_view,
    vecmem::data::vector_view<typename candidate_link::link_index_type>
        tips_view,
    unsigned int& n_out_params) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    device::propagate_to_next_surface<propagator_t, config_t>(
        gid, cfg, det_data, nav_candidates_buffer, in_params_view, links_view,
        step, n_candidates, out_params_view, param_to_link_view, tips_view,
        n_out_params);
}

/// CUDA kernel for running @c traccc::device::build_tracks
__global__ void build_tracks(
    measurement_container_types::const_view measurements_view,
    bound_track_parameters_collection_types::const_view seeds_view,
    vecmem::data::jagged_vector_view<const candidate_link> links_view,
    vecmem::data::jagged_vector_view<const unsigned int> param_to_link_view,
    vecmem::data::vector_view<const typename candidate_link::link_index_type>
        tips_view,
    track_candidate_container_types::view track_candidates_view) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    device::build_tracks(gid, measurements_view, seeds_view, links_view,
                         param_to_link_view, tips_view, track_candidates_view);
}

}  // namespace kernels

template <typename stepper_t, typename navigator_t>
finding_algorithm<stepper_t, navigator_t>::finding_algorithm(
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
finding_algorithm<stepper_t, navigator_t>::operator()(
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

    // Create a map for links
    std::map<unsigned int, vecmem::data::vector_buffer<candidate_link>>
        link_map;

    // Create a map for parameter ID to link ID
    std::map<unsigned int, vecmem::data::vector_buffer<unsigned int>>
        param_to_link_map;

    // Create a map for tip links
    std::map<unsigned int, vecmem::data::vector_buffer<
                               typename candidate_link::link_index_type>>
        tips_map;

    // Link size
    std::vector<std::size_t> n_candidates_per_step;
    n_candidates_per_step.reserve(m_cfg.max_track_candidates_per_track);

    std::vector<std::size_t> n_parameters_per_step;
    n_parameters_per_step.reserve(m_cfg.max_track_candidates_per_track);

    // Global counter object in Device memory
    vecmem::unique_alloc_ptr<device::finding_global_counter>
        global_counter_device =
            vecmem::make_unique_alloc<device::finding_global_counter>(
                m_mr.main);

    // Global counter object in Host memory
    device::finding_global_counter global_counter_host;

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

    for (unsigned int step = 0; step < m_cfg.max_track_candidates_per_track;
         step++) {

        // Global counter object: Device -> Host
        CUDA_ERROR_CHECK(cudaMemcpy(
            &global_counter_host, global_counter_device.get(),
            sizeof(device::finding_global_counter), cudaMemcpyDeviceToHost));

        // Set the number of input parameters
        const unsigned int n_in_params = (step == 0)
                                             ? in_params_buffer.size()
                                             : global_counter_host.n_out_params;

        // Terminate if there is no parameter to process.
        if (n_in_params == 0) {
            break;
        }

        // Reset the global counter
        CUDA_ERROR_CHECK(cudaMemset(global_counter_device.get(), 0,
                                    sizeof(device::finding_global_counter)));

        /*****************************************************************
         * Kernel2: Apply material interaction
         ****************************************************************/

        nThreads = WARP_SIZE * 2;
        nBlocks = (n_in_params + nThreads - 1) / nThreads;
        kernels::apply_interaction<detector_type><<<nBlocks, nThreads>>>(
            det_view, navigation_buffer, n_in_params, in_params_buffer);
        CUDA_ERROR_CHECK(cudaGetLastError());
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());

        /*****************************************************************
         * Kernel3: Count the number of measurements per parameter
         ****************************************************************/

        // Create a buffer for the number of measurements per parameter
        vecmem::data::vector_buffer<unsigned int> n_measurements_buffer(
            n_in_params, m_mr.main);

        nThreads = WARP_SIZE * 2;
        nBlocks = (n_in_params + nThreads - 1) / nThreads;
        kernels::count_measurements<detector_type><<<nBlocks, nThreads>>>(
            det_view, measurements, module_map_buffer, n_in_params,
            in_params_buffer, n_measurements_buffer,
            (*global_counter_device).n_total_measurements);
        CUDA_ERROR_CHECK(cudaGetLastError());
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());

        /*******************************************************************
         * Kernel4: Get Prefix sum on number of threads for each parameter
         *******************************************************************/

        // Global counter object: Device -> Host
        CUDA_ERROR_CHECK(cudaMemcpy(
            &global_counter_host, global_counter_device.get(),
            sizeof(device::finding_global_counter), cudaMemcpyDeviceToHost));

        // Create a buffer for the number of threads per parameter
        vecmem::data::vector_buffer<unsigned int> n_threads_buffer(n_in_params,
                                                                   m_mr.main);

        nThreads = WARP_SIZE * 2;
        nBlocks = (n_in_params + nThreads - 1) / nThreads;
        kernels::count_threads<<<nBlocks, nThreads>>>(
            m_cfg, n_measurements_buffer,
            (*global_counter_device).n_total_measurements, n_threads_buffer,
            (*global_counter_device).n_measurements_per_thread,
            (*global_counter_device).n_total_threads);
        CUDA_ERROR_CHECK(cudaGetLastError());
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());

        // Get Prefix Sum of n_thread vector
        vecmem::device_vector<unsigned int> n_threads(n_threads_buffer);
        thrust::inclusive_scan(thrust::device, n_threads.begin(),
                               n_threads.end(), n_threads.begin());

        /*****************************************************************
         * Kernel5: Find valid tracks
         *****************************************************************/

        // Global counter object: Device -> Host
        CUDA_ERROR_CHECK(cudaMemcpy(
            &global_counter_host, global_counter_device.get(),
            sizeof(device::finding_global_counter), cudaMemcpyDeviceToHost));

        // Buffer for kalman-updated parameters spawned by the measurement
        // candidates
        bound_track_parameters_collection_types::buffer updated_params_buffer(
            n_in_params * m_cfg.max_num_branches_per_surface, m_mr.main);

        // Create the link map
        link_map[step] = {n_in_params * m_cfg.max_num_branches_per_surface,
                          m_mr.main};
        m_copy->setup(link_map[step]);

        nThreads = WARP_SIZE * 2;
        nBlocks =
            (global_counter_host.n_total_threads + nThreads - 1) / nThreads;
        kernels::find_tracks<detector_type, config_type><<<nBlocks, nThreads>>>(
            m_cfg, det_view, measurements, module_map_buffer, in_params_buffer,
            n_threads_buffer, step,
            (*global_counter_device).n_measurements_per_thread,
            (*global_counter_device).n_total_threads, updated_params_buffer,
            link_map[step], (*global_counter_device).n_candidates);
        CUDA_ERROR_CHECK(cudaGetLastError());
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());

        /*****************************************************************
         * Kernel6: Propagate to the next surface
         *****************************************************************/

        // Global counter object: Device -> Host
        CUDA_ERROR_CHECK(cudaMemcpy(
            &global_counter_host, global_counter_device.get(),
            sizeof(device::finding_global_counter), cudaMemcpyDeviceToHost));

        // Buffer for out parameters for the next step
        bound_track_parameters_collection_types::buffer out_params_buffer(
            global_counter_host.n_candidates, m_mr.main);

        // Create the param to link ID map
        param_to_link_map[step] = {global_counter_host.n_candidates, m_mr.main};
        m_copy->setup(param_to_link_map[step]);

        // Create the tip map
        tips_map[step] = {global_counter_host.n_candidates, m_mr.main,
                          vecmem::data::buffer_type::resizable};
        m_copy->setup(tips_map[step]);

        nThreads = WARP_SIZE * 2;

        if (global_counter_host.n_candidates > 0) {
            nBlocks =
                (global_counter_host.n_candidates + nThreads - 1) / nThreads;
            kernels::propagate_to_next_surface<propagator_type, config_type>
                <<<nBlocks, nThreads>>>(
                    m_cfg, det_view, navigation_buffer, updated_params_buffer,
                    link_map[step], step, (*global_counter_device).n_candidates,
                    out_params_buffer, param_to_link_map[step], tips_map[step],
                    (*global_counter_device).n_out_params);
            CUDA_ERROR_CHECK(cudaGetLastError());
            CUDA_ERROR_CHECK(cudaDeviceSynchronize());
        }

        // Global counter object: Device -> Host
        CUDA_ERROR_CHECK(cudaMemcpy(
            &global_counter_host, global_counter_device.get(),
            sizeof(device::finding_global_counter), cudaMemcpyDeviceToHost));

        // Fill the candidate size vector
        n_candidates_per_step.push_back(global_counter_host.n_candidates);
        n_parameters_per_step.push_back(global_counter_host.n_out_params);

        // Swap parameter buffer for the next step
        in_params_buffer = std::move(out_params_buffer);
    }

    // Create link buffer
    vecmem::data::jagged_vector_buffer<candidate_link> links_buffer(
        n_candidates_per_step, m_mr.main, m_mr.host);
    m_copy->setup(links_buffer);

    // Copy link map to link buffer
    const auto n_steps = n_candidates_per_step.size();
    for (unsigned int it = 0; it < n_steps; it++) {

        vecmem::device_vector<candidate_link> in(link_map[it]);
        vecmem::device_vector<candidate_link> out(
            *(links_buffer.host_ptr() + it));

        thrust::copy(thrust::device, in.begin(),
                     in.begin() + n_candidates_per_step[it], out.begin());
    }

    // Create param_to_link
    vecmem::data::jagged_vector_buffer<unsigned int> param_to_link_buffer(
        n_parameters_per_step, m_mr.main, m_mr.host);
    m_copy->setup(param_to_link_buffer);

    // Copy param_to_link map to param_to_link buffer
    for (unsigned int it = 0; it < n_steps; it++) {

        vecmem::device_vector<unsigned int> in(param_to_link_map[it]);
        vecmem::device_vector<unsigned int> out(
            *(param_to_link_buffer.host_ptr() + it));

        thrust::copy(thrust::device, in.begin(),
                     in.begin() + n_parameters_per_step[it], out.begin());
    }

    // Get the number of tips per step
    std::vector<unsigned int> n_tips_per_step;
    n_tips_per_step.reserve(n_steps);
    for (unsigned int it = 0; it < n_steps; it++) {
        n_tips_per_step.push_back(m_copy->get_size(tips_map[it]));
    }

    // Copy tips_map into the tips vector (D->D)
    unsigned int n_tips_total =
        std::accumulate(n_tips_per_step.begin(), n_tips_per_step.end(), 0);
    vecmem::data::vector_buffer<typename candidate_link::link_index_type>
        tips_buffer{n_tips_total, m_mr.main};
    m_copy->setup(tips_buffer);

    vecmem::device_vector<typename candidate_link::link_index_type> tips(
        tips_buffer);

    unsigned int prefix_sum = 0;
    for (unsigned int it = m_cfg.min_track_candidates_per_track - 1;
         it < n_steps; it++) {

        vecmem::device_vector<typename candidate_link::link_index_type> in(
            tips_map[it]);

        const unsigned int n_tips = n_tips_per_step[it];
        if (n_tips > 0) {
            thrust::copy(thrust::device, in.begin(), in.begin() + n_tips,
                         tips.begin() + prefix_sum);
            prefix_sum += n_tips;
        }
    }

    /*****************************************************************
     * Kernel7: Build tracks
     *****************************************************************/

    // Create track candidate buffer
    track_candidate_container_types::buffer track_candidates_buffer{
        {n_tips_total, m_mr.main},
        {std::vector<std::size_t>(n_tips_total,
                                  m_cfg.max_track_candidates_per_track),
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

// Explicit template instantiation
using toy_detector_type =
    detray::detector<detray::detector_registry::toy_detector,
                     covfie::field_view, detray::device_container_types>;
using toy_stepper_type = detray::rk_stepper<
    covfie::field<toy_detector_type::bfield_backend_type>::view_t, transform3,
    detray::constrained_step<>>;
using toy_navigator_type = detray::navigator<const toy_detector_type>;
template class finding_algorithm<toy_stepper_type, toy_navigator_type>;

using device_detector_type =
    detray::detector<detray::detector_registry::template telescope_detector<
                         detray::rectangle2D<>>,
                     covfie::field_view, detray::device_container_types>;
using rk_stepper_type = detray::rk_stepper<
    covfie::field<device_detector_type::bfield_backend_type>::view_t,
    transform3, detray::constrained_step<>>;
using device_navigator_type = detray::navigator<const device_detector_type>;
template class finding_algorithm<rk_stepper_type, device_navigator_type>;

}  // namespace traccc::cuda
