/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "../sanity/contiguous_on.cuh"
#include "../utils/cuda_error_handling.hpp"
#include "../utils/utils.hpp"
#include "traccc/cuda/finding/finding_algorithm.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/device/finding_global_counter.hpp"
#include "traccc/finding/candidate_link.hpp"
#include "traccc/finding/device/add_links_for_holes.hpp"
#include "traccc/finding/device/apply_interaction.hpp"
#include "traccc/finding/device/build_tracks.hpp"
#include "traccc/finding/device/count_measurements.hpp"
#include "traccc/finding/device/find_tracks.hpp"
#include "traccc/finding/device/make_barcode_sequence.hpp"
#include "traccc/finding/device/propagate_to_next_surface.hpp"
#include "traccc/finding/device/prune_tracks.hpp"
#include "traccc/utils/projections.hpp"

// detray include(s).
#include "detray/core/detector.hpp"
#include "detray/core/detector_metadata.hpp"
#include "detray/detectors/bfield.hpp"
#include "detray/navigation/navigator.hpp"
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
#include <thrust/unique.h>

// System include(s).
#include <cassert>
#include <vector>

namespace traccc::cuda {
namespace kernels {

/// CUDA kernel for running @c traccc::device::make_barcode_sequence
__global__ void make_barcode_sequence(
    measurement_collection_types::const_view measurements_view,
    vecmem::data::vector_view<detray::geometry::barcode> barcodes_view) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    device::make_barcode_sequence(gid, measurements_view, barcodes_view);
}

/// CUDA kernel for running @c traccc::device::apply_interaction
template <typename detector_t>
__global__ void apply_interaction(
    typename detector_t::view_type det_data, const finding_config cfg,
    const int n_params,
    bound_track_parameters_collection_types::view params_view) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    device::apply_interaction<detector_t>(gid, cfg, det_data, n_params,
                                          params_view);
}

/// CUDA kernel for running @c traccc::device::count_measurements
__global__ void count_measurements(
    bound_track_parameters_collection_types::const_view params_view,
    vecmem::data::vector_view<const detray::geometry::barcode> barcodes_view,
    vecmem::data::vector_view<const unsigned int> upper_bounds_view,
    const unsigned int n_in_params,
    vecmem::data::vector_view<unsigned int> n_measurements_view,
    vecmem::data::vector_view<unsigned int> ref_meas_idx_view,
    unsigned int& n_measurements_sum) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    device::count_measurements(
        gid, params_view, barcodes_view, upper_bounds_view, n_in_params,
        n_measurements_view, ref_meas_idx_view, n_measurements_sum);
}

/// CUDA kernel for running @c traccc::device::find_tracks
template <typename detector_t, typename config_t>
__global__ void find_tracks(
    const config_t cfg, typename detector_t::view_type det_data,
    measurement_collection_types::const_view measurements_view,
    bound_track_parameters_collection_types::const_view in_params_view,
    vecmem::data::vector_view<const unsigned int>
        n_measurements_prefix_sum_view,
    vecmem::data::vector_view<const unsigned int> ref_meas_idx_view,
    vecmem::data::vector_view<const candidate_link> prev_links_view,
    vecmem::data::vector_view<const unsigned int> prev_param_to_link_view,
    const unsigned int step, const unsigned int n_max_candidates,
    bound_track_parameters_collection_types::view out_params_view,
    vecmem::data::vector_view<unsigned int> n_candidates_view,
    vecmem::data::vector_view<candidate_link> links_view,
    unsigned int& n_candidates) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    device::find_tracks<detector_t, config_t>(
        gid, cfg, det_data, measurements_view, in_params_view,
        n_measurements_prefix_sum_view, ref_meas_idx_view, prev_links_view,
        prev_param_to_link_view, step, n_max_candidates, out_params_view,
        n_candidates_view, links_view, n_candidates);
}

/// CUDA kernel for running @c traccc::device::add_links_for_holes
__global__ void add_links_for_holes(
    vecmem::data::vector_view<const unsigned int> n_candidates_view,
    bound_track_parameters_collection_types::const_view in_params_view,
    vecmem::data::vector_view<const candidate_link> prev_links_view,
    vecmem::data::vector_view<const unsigned int> prev_param_to_link_view,
    const unsigned int step, const unsigned int n_max_candidates,
    bound_track_parameters_collection_types::view out_params_view,
    vecmem::data::vector_view<candidate_link> links_view,
    unsigned int& n_total_candidates) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    device::add_links_for_holes(gid, n_candidates_view, in_params_view,
                                prev_links_view, prev_param_to_link_view, step,
                                n_max_candidates, out_params_view, links_view,
                                n_total_candidates);
}

/// CUDA kernel for running @c traccc::device::propagate_to_next_surface
template <typename propagator_t, typename bfield_t, typename config_t>
__global__ void propagate_to_next_surface(
    const config_t cfg,
    typename propagator_t::detector_type::view_type det_data,
    bfield_t field_data,
    vecmem::data::jagged_vector_view<typename propagator_t::intersection_type>
        nav_candidates_buffer,
    bound_track_parameters_collection_types::const_view in_params_view,
    vecmem::data::vector_view<const candidate_link> links_view,
    const unsigned int step, const unsigned int& n_candidates,
    bound_track_parameters_collection_types::view out_params_view,
    vecmem::data::vector_view<unsigned int> param_to_link_view,
    vecmem::data::vector_view<typename candidate_link::link_index_type>
        tips_view,
    vecmem::data::vector_view<unsigned int> n_tracks_per_seed_view,
    unsigned int& n_out_params) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    device::propagate_to_next_surface<propagator_t, bfield_t, config_t>(
        gid, cfg, det_data, field_data, nav_candidates_buffer, in_params_view,
        links_view, step, n_candidates, out_params_view, param_to_link_view,
        tips_view, n_tracks_per_seed_view, n_out_params);
}

/// CUDA kernel for running @c traccc::device::build_tracks
template <typename config_t>
__global__ void build_tracks(
    const config_t cfg,
    measurement_collection_types::const_view measurements_view,
    bound_track_parameters_collection_types::const_view seeds_view,
    vecmem::data::jagged_vector_view<const candidate_link> links_view,
    vecmem::data::jagged_vector_view<const unsigned int> param_to_link_view,
    vecmem::data::vector_view<const typename candidate_link::link_index_type>
        tips_view,
    track_candidate_container_types::view track_candidates_view,
    vecmem::data::vector_view<unsigned int> valid_indices_view,
    unsigned int& n_valid_tracks) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    device::build_tracks(gid, cfg, measurements_view, seeds_view, links_view,
                         param_to_link_view, tips_view, track_candidates_view,
                         valid_indices_view, n_valid_tracks);
}

/// CUDA kernel for running @c traccc::device::prune_tracks
__global__ void prune_tracks(
    track_candidate_container_types::const_view track_candidates_view,
    vecmem::data::vector_view<const unsigned int> valid_indices_view,
    track_candidate_container_types::view prune_candidates_view) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    device::prune_tracks(gid, track_candidates_view, valid_indices_view,
                         prune_candidates_view);
}

}  // namespace kernels

template <typename stepper_t, typename navigator_t>
finding_algorithm<stepper_t, navigator_t>::finding_algorithm(
    const config_type& cfg, const traccc::memory_resource& mr,
    vecmem::copy& copy, stream& str)
    : m_cfg(cfg),
      m_mr(mr),
      m_copy(copy),
      m_stream(str),
      m_warp_size(details::get_warp_size(str.device())) {}

template <typename stepper_t, typename navigator_t>
track_candidate_container_types::buffer
finding_algorithm<stepper_t, navigator_t>::operator()(
    const typename detector_type::view_type& det_view,
    const bfield_type& field_view,
    const vecmem::data::jagged_vector_view<
        typename navigator_t::intersection_type>& navigation_buffer,
    const typename measurement_collection_types::view& measurements,
    const bound_track_parameters_collection_types::buffer& seeds_buffer) const {

    // Get a convenience variable for the stream that we'll be using.
    cudaStream_t stream = details::get_stream(m_stream);

    // Copy setup
    m_copy.setup(seeds_buffer);
    m_copy.setup(navigation_buffer);

    const unsigned int n_seeds = m_copy.get_size(seeds_buffer);

    // Prepare input parameters with seeds
    bound_track_parameters_collection_types::buffer in_params_buffer(n_seeds,
                                                                     m_mr.main);
    bound_track_parameters_collection_types::device in_params(in_params_buffer);
    bound_track_parameters_collection_types::const_device seeds(seeds_buffer);
    thrust::copy(thrust::cuda::par.on(stream), seeds.begin(), seeds.end(),
                 in_params.begin());

    // Number of tracks per seed
    vecmem::data::vector_buffer<unsigned int> n_tracks_per_seed_buffer(
        n_seeds, m_mr.main);

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
     * Measurement Operations
     *****************************************************************/

    measurement_collection_types::const_view::size_type n_measurements =
        m_copy.get_size(measurements);

    assert(is_contiguous_on(measurement_module_projection(), m_mr.main, m_copy,
                            m_stream, measurements));

    // Get copy of barcode uniques
    measurement_collection_types::buffer uniques_buffer{n_measurements,
                                                        m_mr.main};
    measurement_collection_types::device uniques(uniques_buffer);

    measurement* end =
        thrust::unique_copy(thrust::cuda::par.on(stream), measurements.ptr(),
                            measurements.ptr() + n_measurements,
                            uniques.begin(), measurement_equal_comp());
    unsigned int n_modules = end - uniques.begin();

    // Get upper bounds of unique elements
    vecmem::data::vector_buffer<unsigned int> upper_bounds_buffer{n_modules,
                                                                  m_mr.main};
    vecmem::device_vector<unsigned int> upper_bounds(upper_bounds_buffer);

    thrust::upper_bound(thrust::cuda::par.on(stream), measurements.ptr(),
                        measurements.ptr() + n_measurements, uniques.begin(),
                        uniques.begin() + n_modules, upper_bounds.begin(),
                        measurement_sort_comp());

    /*****************************************************************
     * Kernel1: Create barcode sequence
     *****************************************************************/

    vecmem::data::vector_buffer<detray::geometry::barcode> barcodes_buffer{
        n_modules, m_mr.main};

    unsigned int nThreads = m_warp_size * 2;
    unsigned int nBlocks = (barcodes_buffer.size() + nThreads - 1) / nThreads;

    kernels::make_barcode_sequence<<<nBlocks, nThreads, 0, stream>>>(
        uniques_buffer, barcodes_buffer);

    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

    for (unsigned int step = 0; step < m_cfg.max_track_candidates_per_track;
         step++) {

        // Previous step
        const unsigned int prev_step = (step == 0 ? 0 : step - 1);

        // Reset the number of tracks per seed
        m_copy.memset(n_tracks_per_seed_buffer, 0)->ignore();

        // Global counter object: Device -> Host
        TRACCC_CUDA_ERROR_CHECK(
            cudaMemcpyAsync(&global_counter_host, global_counter_device.get(),
                            sizeof(device::finding_global_counter),
                            cudaMemcpyDeviceToHost, stream));

        m_stream.synchronize();

        // Set the number of input parameters
        const unsigned int n_in_params = (step == 0)
                                             ? in_params_buffer.size()
                                             : global_counter_host.n_out_params;

        // Terminate if there is no parameter to process.
        if (n_in_params == 0) {
            break;
        }

        // Reset the global counter
        TRACCC_CUDA_ERROR_CHECK(
            cudaMemsetAsync(global_counter_device.get(), 0,
                            sizeof(device::finding_global_counter), stream));

        /*****************************************************************
         * Kernel2: Apply material interaction
         ****************************************************************/

        nThreads = m_warp_size * 2;
        nBlocks = (n_in_params + nThreads - 1) / nThreads;
        kernels::apply_interaction<detector_type>
            <<<nBlocks, nThreads, 0, stream>>>(det_view, m_cfg, n_in_params,
                                               in_params_buffer);
        TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

        /*****************************************************************
         * Kernel3: Count the number of measurements per parameter
         ****************************************************************/

        vecmem::data::vector_buffer<unsigned int> n_measurements_buffer(
            n_in_params, m_mr.main);
        vecmem::device_vector<unsigned int> n_measurements_device(
            n_measurements_buffer);
        thrust::fill(thrust::cuda::par.on(stream),
                     n_measurements_device.begin(), n_measurements_device.end(),
                     0u);

        // Create a buffer for the first measurement index of parameter
        vecmem::data::vector_buffer<unsigned int> ref_meas_idx_buffer(
            n_in_params, m_mr.main);

        nThreads = m_warp_size * 2;
        nBlocks = (n_in_params + nThreads - 1) / nThreads;
        kernels::count_measurements<<<nBlocks, nThreads, 0, stream>>>(
            in_params_buffer, barcodes_buffer, upper_bounds_buffer, n_in_params,
            n_measurements_buffer, ref_meas_idx_buffer,
            (*global_counter_device).n_measurements_sum);
        TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

        // Global counter object: Device -> Host
        TRACCC_CUDA_ERROR_CHECK(
            cudaMemcpyAsync(&global_counter_host, global_counter_device.get(),
                            sizeof(device::finding_global_counter),
                            cudaMemcpyDeviceToHost, stream));

        m_stream.synchronize();

        // Create the buffer for the prefix sum of the number of measurements
        // per parameter
        vecmem::data::vector_buffer<unsigned int>
            n_measurements_prefix_sum_buffer(n_in_params, m_mr.main);
        vecmem::device_vector<unsigned int> n_measurements_prefix_sum(
            n_measurements_prefix_sum_buffer);
        thrust::inclusive_scan(
            thrust::cuda::par.on(stream), n_measurements_device.begin(),
            n_measurements_device.end(), n_measurements_prefix_sum.begin());

        /*****************************************************************
         * Kernel4: Find valid tracks
         *****************************************************************/

        // Buffer for kalman-updated parameters spawned by the measurement
        // candidates
        const unsigned int n_max_candidates =
            n_in_params * m_cfg.max_num_branches_per_surface;

        vecmem::data::vector_buffer<unsigned int> n_candidates_buffer{
            n_in_params, m_mr.main};
        vecmem::device_vector<unsigned int> n_candidates_device(
            n_candidates_buffer);
        thrust::fill(thrust::cuda::par.on(stream), n_candidates_device.begin(),
                     n_candidates_device.end(), 0u);

        bound_track_parameters_collection_types::buffer updated_params_buffer(
            n_in_params * m_cfg.max_num_branches_per_surface, m_mr.main);

        // Create the link map
        link_map[step] = {n_in_params * m_cfg.max_num_branches_per_surface,
                          m_mr.main};
        m_copy.setup(link_map[step]);
        nBlocks = (global_counter_host.n_measurements_sum +
                   nThreads * m_cfg.n_measurements_per_thread - 1) /
                  (nThreads * m_cfg.n_measurements_per_thread);

        if (nBlocks > 0) {
            kernels::find_tracks<detector_type, config_type>
                <<<nBlocks, nThreads, 0, stream>>>(
                    m_cfg, det_view, measurements, in_params_buffer,
                    n_measurements_prefix_sum_buffer, ref_meas_idx_buffer,
                    link_map[prev_step], param_to_link_map[prev_step], step,
                    n_max_candidates, updated_params_buffer,
                    n_candidates_buffer, link_map[step],
                    (*global_counter_device).n_candidates);
            TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
        }

        /*****************************************************************
         * Kernel5: Add a dummy links in case of no branches
         *****************************************************************/

        nBlocks = (n_in_params + nThreads - 1) / nThreads;

        if (nBlocks > 0) {
            kernels::add_links_for_holes<<<nBlocks, nThreads, 0, stream>>>(
                n_candidates_buffer, in_params_buffer, link_map[prev_step],
                param_to_link_map[prev_step], step, n_max_candidates,
                updated_params_buffer, link_map[step],
                (*global_counter_device).n_candidates);
            TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
        }

        // Global counter object: Device -> Host
        TRACCC_CUDA_ERROR_CHECK(
            cudaMemcpyAsync(&global_counter_host, global_counter_device.get(),
                            sizeof(device::finding_global_counter),
                            cudaMemcpyDeviceToHost, stream));

        m_stream.synchronize();

        /*****************************************************************
         * Kernel6: Propagate to the next surface
         *****************************************************************/

        // Buffer for out parameters for the next step
        bound_track_parameters_collection_types::buffer out_params_buffer(
            global_counter_host.n_candidates, m_mr.main);

        // Create the param to link ID map
        param_to_link_map[step] = {global_counter_host.n_candidates, m_mr.main};
        m_copy.setup(param_to_link_map[step]);

        // Create the tip map
        tips_map[step] = {global_counter_host.n_candidates, m_mr.main,
                          vecmem::data::buffer_type::resizable};
        m_copy.setup(tips_map[step]);

        nThreads = m_warp_size * 2;

        if (global_counter_host.n_candidates > 0) {
            nBlocks =
                (global_counter_host.n_candidates + nThreads - 1) / nThreads;
            kernels::propagate_to_next_surface<propagator_type, bfield_type,
                                               config_type>
                <<<nBlocks, nThreads, 0, stream>>>(
                    m_cfg, det_view, field_view, navigation_buffer,
                    updated_params_buffer, link_map[step], step,
                    (*global_counter_device).n_candidates, out_params_buffer,
                    param_to_link_map[step], tips_map[step],
                    n_tracks_per_seed_buffer,
                    (*global_counter_device).n_out_params);
            TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
        }

        TRACCC_CUDA_ERROR_CHECK(
            cudaMemcpyAsync(&global_counter_host, global_counter_device.get(),
                            sizeof(device::finding_global_counter),
                            cudaMemcpyDeviceToHost, stream));

        m_stream.synchronize();

        // Fill the candidate size vector
        n_candidates_per_step.push_back(global_counter_host.n_candidates);
        n_parameters_per_step.push_back(global_counter_host.n_out_params);

        // Swap parameter buffer for the next step
        in_params_buffer = std::move(out_params_buffer);
    }

    // Create link buffer
    vecmem::data::jagged_vector_buffer<candidate_link> links_buffer(
        n_candidates_per_step, m_mr.main, m_mr.host);
    m_copy.setup(links_buffer);

    // Copy link map to link buffer
    const auto n_steps = n_candidates_per_step.size();
    for (unsigned int it = 0; it < n_steps; it++) {

        vecmem::device_vector<candidate_link> in(link_map[it]);
        vecmem::device_vector<candidate_link> out(
            *(links_buffer.host_ptr() + it));

        thrust::copy(thrust::cuda::par.on(stream), in.begin(),
                     in.begin() + n_candidates_per_step[it], out.begin());
    }

    // Create param_to_link
    vecmem::data::jagged_vector_buffer<unsigned int> param_to_link_buffer(
        n_parameters_per_step, m_mr.main, m_mr.host);
    m_copy.setup(param_to_link_buffer);

    // Copy param_to_link map to param_to_link buffer
    for (unsigned int it = 0; it < n_steps; it++) {

        vecmem::device_vector<unsigned int> in(param_to_link_map[it]);
        vecmem::device_vector<unsigned int> out(
            *(param_to_link_buffer.host_ptr() + it));

        thrust::copy(thrust::cuda::par.on(stream), in.begin(),
                     in.begin() + n_parameters_per_step[it], out.begin());
    }

    // Get the number of tips per step
    std::vector<unsigned int> n_tips_per_step;
    n_tips_per_step.reserve(n_steps);
    for (unsigned int it = 0; it < n_steps; it++) {
        n_tips_per_step.push_back(m_copy.get_size(tips_map[it]));
    }

    // Copy tips_map into the tips vector (D->D)
    unsigned int n_tips_total =
        std::accumulate(n_tips_per_step.begin(), n_tips_per_step.end(), 0);
    vecmem::data::vector_buffer<typename candidate_link::link_index_type>
        tips_buffer{n_tips_total, m_mr.main};
    m_copy.setup(tips_buffer);

    vecmem::device_vector<typename candidate_link::link_index_type> tips(
        tips_buffer);

    unsigned int prefix_sum = 0;

    for (unsigned int it = 0; it < n_steps; it++) {
        vecmem::device_vector<typename candidate_link::link_index_type> in(
            tips_map[it]);

        const unsigned int n_tips = n_tips_per_step[it];
        if (n_tips > 0) {
            thrust::copy(thrust::cuda::par.on(stream), in.begin(),
                         in.begin() + n_tips, tips.begin() + prefix_sum);
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

    m_copy.setup(track_candidates_buffer.headers);
    m_copy.setup(track_candidates_buffer.items);

    // Create buffer for valid indices
    vecmem::data::vector_buffer<unsigned int> valid_indices_buffer(n_tips_total,
                                                                   m_mr.main);

    // @Note: nBlocks can be zero in case there is no tip. This happens when
    // chi2_max config is set tightly and no tips are found
    if (n_tips_total > 0) {
        nThreads = m_warp_size * 2;
        nBlocks = (n_tips_total + nThreads - 1) / nThreads;
        kernels::build_tracks<<<nBlocks, nThreads, 0, stream>>>(
            m_cfg, measurements, seeds_buffer, links_buffer,
            param_to_link_buffer, tips_buffer, track_candidates_buffer,
            valid_indices_buffer, (*global_counter_device).n_valid_tracks);
        TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
    }

    // Global counter object: Device -> Host
    TRACCC_CUDA_ERROR_CHECK(
        cudaMemcpyAsync(&global_counter_host, global_counter_device.get(),
                        sizeof(device::finding_global_counter),
                        cudaMemcpyDeviceToHost, stream));

    m_stream.synchronize();

    // Create pruned candidate buffer
    track_candidate_container_types::buffer prune_candidates_buffer{
        {global_counter_host.n_valid_tracks, m_mr.main},
        {std::vector<std::size_t>(global_counter_host.n_valid_tracks,
                                  m_cfg.max_track_candidates_per_track),
         m_mr.main, m_mr.host, vecmem::data::buffer_type::resizable}};

    m_copy.setup(prune_candidates_buffer.headers);
    m_copy.setup(prune_candidates_buffer.items);

    if (global_counter_host.n_valid_tracks > 0) {
        nThreads = m_warp_size * 2;
        nBlocks =
            (global_counter_host.n_valid_tracks + nThreads - 1) / nThreads;

        kernels::prune_tracks<<<nBlocks, nThreads, 0, stream>>>(
            track_candidates_buffer, valid_indices_buffer,
            prune_candidates_buffer);
        TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
    }

    return prune_candidates_buffer;
}

// Explicit template instantiation
using default_detector_type =
    detray::detector<detray::default_metadata, detray::device_container_types>;
using default_stepper_type =
    detray::rk_stepper<covfie::field<detray::bfield::const_bknd_t>::view_t,
                       traccc::default_algebra, detray::constrained_step<>>;
using default_navigator_type = detray::navigator<const default_detector_type>;
template class finding_algorithm<default_stepper_type, default_navigator_type>;

}  // namespace traccc::cuda
