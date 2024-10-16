/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "../sanity/contiguous_on.cuh"
#include "../utils/barrier.hpp"
#include "../utils/cuda_error_handling.hpp"
#include "../utils/utils.hpp"
#include "traccc/cuda/finding/finding_algorithm.hpp"
#include "traccc/cuda/utils/thread_id.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/device/sort_key.hpp"
#include "traccc/finding/candidate_link.hpp"
#include "traccc/finding/device/apply_interaction.hpp"
#include "traccc/finding/device/build_tracks.hpp"
#include "traccc/finding/device/fill_sort_keys.hpp"
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
#include "vecmem/containers/data/vector_view.hpp"
#include "vecmem/memory/unique_ptr.hpp"

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
    bound_track_parameters_collection_types::view params_view,
    vecmem::data::vector_view<const unsigned int> params_liveness_view) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    device::apply_interaction<detector_t>(gid, cfg, det_data, n_params,
                                          params_view, params_liveness_view);
}

/// CUDA kernel for running @c traccc::device::find_tracks
template <typename detector_t, typename config_t>
__global__ void find_tracks(
    const config_t cfg, typename detector_t::view_type det_data,
    measurement_collection_types::const_view measurements_view,
    bound_track_parameters_collection_types::const_view in_params_view,
    vecmem::data::vector_view<const unsigned int> in_params_liveness_view,
    const unsigned int n_in_params,
    vecmem::data::vector_view<const detray::geometry::barcode> barcodes_view,
    vecmem::data::vector_view<const unsigned int> upper_bounds_view,
    vecmem::data::vector_view<const candidate_link> prev_links_view,
    const unsigned int step, const unsigned int n_max_candidates,
    bound_track_parameters_collection_types::view out_params_view,
    vecmem::data::vector_view<unsigned int> out_params_liveness_view,
    vecmem::data::vector_view<candidate_link> links_view,
    unsigned int* n_candidates) {
    __shared__ unsigned int shared_candidates_size;
    extern __shared__ unsigned int s[];
    unsigned int* shared_num_candidates = s;
    std::pair<unsigned int, unsigned int>* shared_candidates =
        reinterpret_cast<std::pair<unsigned int, unsigned int>*>(
            &shared_num_candidates[blockDim.x]);

    cuda::barrier barrier;
    cuda::thread_id1 thread_id;

    device::find_tracks<cuda::thread_id1, cuda::barrier, detector_t, config_t>(
        thread_id, barrier, cfg, det_data, measurements_view, in_params_view,
        in_params_liveness_view, n_in_params, barcodes_view, upper_bounds_view,
        prev_links_view, step, n_max_candidates, out_params_view,
        out_params_liveness_view, links_view, *n_candidates,
        shared_num_candidates, shared_candidates, shared_candidates_size);
}

/// CUDA kernel for running @c traccc::device::fill_sort_keys
__global__ void fill_sort_keys(
    bound_track_parameters_collection_types::const_view params_view,
    vecmem::data::vector_view<device::sort_key> keys_view,
    vecmem::data::vector_view<unsigned int> ids_view) {

    device::fill_sort_keys(threadIdx.x + blockIdx.x * blockDim.x, params_view,
                           keys_view, ids_view);
}

/// CUDA kernel for running @c traccc::device::propagate_to_next_surface
template <typename propagator_t, typename bfield_t, typename config_t>
__global__ void propagate_to_next_surface(
    const config_t cfg,
    typename propagator_t::detector_type::view_type det_data,
    bfield_t field_data,
    bound_track_parameters_collection_types::view params_view,
    vecmem::data::vector_view<unsigned int> params_liveness_view,
    vecmem::data::vector_view<const unsigned int> param_ids_view,
    vecmem::data::vector_view<const candidate_link> links_view,
    const unsigned int step, const unsigned int n_candidates,
    vecmem::data::vector_view<typename candidate_link::link_index_type>
        tips_view,
    vecmem::data::vector_view<unsigned int> n_tracks_per_seed_view) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    device::propagate_to_next_surface<propagator_t, bfield_t, config_t>(
        gid, cfg, det_data, field_data, params_view, params_liveness_view,
        param_ids_view, links_view, step, n_candidates, tips_view,
        n_tracks_per_seed_view);
}

/// CUDA kernel for running @c traccc::device::build_tracks
template <typename config_t>
__global__ void build_tracks(
    const config_t cfg,
    measurement_collection_types::const_view measurements_view,
    bound_track_parameters_collection_types::const_view seeds_view,
    vecmem::data::jagged_vector_view<const candidate_link> links_view,
    vecmem::data::vector_view<const typename candidate_link::link_index_type>
        tips_view,
    track_candidate_container_types::view track_candidates_view,
    vecmem::data::vector_view<unsigned int> valid_indices_view,
    unsigned int* n_valid_tracks) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    device::build_tracks(gid, cfg, measurements_view, seeds_view, links_view,
                         tips_view, track_candidates_view, valid_indices_view,
                         *n_valid_tracks);
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
    const typename measurement_collection_types::view& measurements,
    const bound_track_parameters_collection_types::buffer& seeds_buffer) const {

    // Get a convenience variable for the stream that we'll be using.
    cudaStream_t stream = details::get_stream(m_stream);

    // Copy setup
    m_copy.setup(seeds_buffer)->ignore();

    /*****************************************************************
     * Measurement Operations
     *****************************************************************/

    unsigned int n_modules;
    measurement_collection_types::const_view::size_type n_measurements =
        m_copy.get_size(measurements);

    // Get copy of barcode uniques
    measurement_collection_types::buffer uniques_buffer{n_measurements,
                                                        m_mr.main};
    m_copy.setup(uniques_buffer)->ignore();

    {
        assert(is_contiguous_on<measurement_collection_types::const_device>(
            measurement_module_projection(), m_mr.main, m_copy, m_stream,
            measurements));

        measurement_collection_types::device uniques(uniques_buffer);

        measurement* uniques_end = thrust::unique_copy(
            thrust::cuda::par.on(stream), measurements.ptr(),
            measurements.ptr() + n_measurements, uniques.begin(),
            measurement_equal_comp());
        n_modules = uniques_end - uniques.begin();
    }

    // Get upper bounds of unique elements
    vecmem::data::vector_buffer<unsigned int> upper_bounds_buffer{n_modules,
                                                                  m_mr.main};
    m_copy.setup(upper_bounds_buffer)->ignore();

    {
        vecmem::device_vector<unsigned int> upper_bounds(upper_bounds_buffer);

        measurement_collection_types::device uniques(uniques_buffer);

        thrust::upper_bound(thrust::cuda::par.on(stream), measurements.ptr(),
                            measurements.ptr() + n_measurements,
                            uniques.begin(), uniques.begin() + n_modules,
                            upper_bounds.begin(), measurement_sort_comp());
    }

    /*****************************************************************
     * Kernel1: Create barcode sequence
     *****************************************************************/

    vecmem::data::vector_buffer<detray::geometry::barcode> barcodes_buffer{
        n_modules, m_mr.main};
    m_copy.setup(barcodes_buffer)->ignore();

    {
        const unsigned int nThreads = m_warp_size * 2;
        const unsigned int nBlocks =
            (barcodes_buffer.size() + nThreads - 1) / nThreads;

        kernels::make_barcode_sequence<<<nBlocks, nThreads, 0, stream>>>(
            uniques_buffer, barcodes_buffer);

        TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
    }

    const unsigned int n_seeds = m_copy.get_size(seeds_buffer);

    // Prepare input parameters with seeds
    bound_track_parameters_collection_types::buffer in_params_buffer(n_seeds,
                                                                     m_mr.main);
    m_copy.setup(in_params_buffer)->ignore();
    m_copy(vecmem::get_data(seeds_buffer), vecmem::get_data(in_params_buffer))
        ->ignore();
    vecmem::data::vector_buffer<unsigned int> param_liveness_buffer(n_seeds,
                                                                    m_mr.main);
    m_copy.setup(param_liveness_buffer)->ignore();
    m_copy.memset(param_liveness_buffer, 1)->ignore();

    // Number of tracks per seed
    vecmem::data::vector_buffer<unsigned int> n_tracks_per_seed_buffer(
        n_seeds, m_mr.main);
    m_copy.setup(n_tracks_per_seed_buffer)->ignore();

    // Create a map for links
    std::map<unsigned int, vecmem::data::vector_buffer<candidate_link>>
        link_map;

    // Create a buffer of tip links
    vecmem::data::vector_buffer<typename candidate_link::link_index_type>
        tips_buffer{m_cfg.max_num_branches_per_seed * n_seeds, m_mr.main,
                    vecmem::data::buffer_type::resizable};
    m_copy.setup(tips_buffer)->wait();

    // Link size
    std::vector<std::size_t> n_candidates_per_step;
    n_candidates_per_step.reserve(m_cfg.max_track_candidates_per_track);

    unsigned int n_in_params = n_seeds;

    for (unsigned int step = 0;
         step < m_cfg.max_track_candidates_per_track && n_in_params > 0;
         step++) {

        /*****************************************************************
         * Kernel2: Apply material interaction
         ****************************************************************/

        {
            const unsigned int nThreads = m_warp_size * 2;
            const unsigned int nBlocks =
                (n_in_params + nThreads - 1) / nThreads;

            kernels::apply_interaction<detector_type>
                <<<nBlocks, nThreads, 0, stream>>>(det_view, m_cfg, n_in_params,
                                                   in_params_buffer,
                                                   param_liveness_buffer);
            TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
        }

        /*****************************************************************
         * Kernel3: Find valid tracks
         *****************************************************************/

        unsigned int n_candidates = 0;

        {
            // Previous step
            const unsigned int prev_step = (step == 0 ? 0 : step - 1);

            // Buffer for kalman-updated parameters spawned by the measurement
            // candidates
            const unsigned int n_max_candidates =
                n_in_params * m_cfg.max_num_branches_per_surface;

            bound_track_parameters_collection_types::buffer
                updated_params_buffer(
                    n_in_params * m_cfg.max_num_branches_per_surface,
                    m_mr.main);
            m_copy.setup(updated_params_buffer)->ignore();

            vecmem::data::vector_buffer<unsigned int> updated_liveness_buffer(
                n_in_params * m_cfg.max_num_branches_per_surface, m_mr.main);
            m_copy.setup(updated_liveness_buffer)->ignore();

            // Create the link map
            link_map[step] = {n_in_params * m_cfg.max_num_branches_per_surface,
                              m_mr.main};
            m_copy.setup(link_map[step])->ignore();

            const unsigned int nThreads = m_warp_size * 2;
            const unsigned int nBlocks =
                (n_in_params + nThreads - 1) / nThreads;

            vecmem::unique_alloc_ptr<unsigned int> n_candidates_device =
                vecmem::make_unique_alloc<unsigned int>(m_mr.main);
            TRACCC_CUDA_ERROR_CHECK(cudaMemsetAsync(
                n_candidates_device.get(), 0, sizeof(unsigned int), stream));

            kernels::find_tracks<detector_type, config_type>
                <<<nBlocks, nThreads,
                   nThreads * sizeof(unsigned int) +
                       2 * nThreads *
                           sizeof(std::pair<unsigned int, unsigned int>),
                   stream>>>(m_cfg, det_view, measurements, in_params_buffer,
                             param_liveness_buffer, n_in_params,
                             barcodes_buffer, upper_bounds_buffer,
                             link_map[prev_step], step, n_max_candidates,
                             updated_params_buffer, updated_liveness_buffer,
                             link_map[step], n_candidates_device.get());
            TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

            std::swap(in_params_buffer, updated_params_buffer);
            std::swap(param_liveness_buffer, updated_liveness_buffer);

            TRACCC_CUDA_ERROR_CHECK(cudaMemcpyAsync(
                &n_candidates, n_candidates_device.get(), sizeof(unsigned int),
                cudaMemcpyDeviceToHost, stream));

            m_stream.synchronize();
        }

        if (n_candidates > 0) {
            /*****************************************************************
             * Kernel4: Get key and value for parameter sorting
             *****************************************************************/

            vecmem::data::vector_buffer<unsigned int> param_ids_buffer(
                n_candidates, m_mr.main);
            m_copy.setup(param_ids_buffer)->ignore();

            {
                vecmem::data::vector_buffer<device::sort_key> keys_buffer(
                    n_candidates, m_mr.main);
                m_copy.setup(keys_buffer)->ignore();

                const unsigned int nThreads = m_warp_size * 2;
                const unsigned int nBlocks =
                    (n_candidates + nThreads - 1) / nThreads;
                kernels::fill_sort_keys<<<nBlocks, nThreads, 0, stream>>>(
                    in_params_buffer, keys_buffer, param_ids_buffer);
                TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

                // Sort the key and values
                vecmem::device_vector<device::sort_key> keys_device(
                    keys_buffer);
                vecmem::device_vector<unsigned int> param_ids_device(
                    param_ids_buffer);
                thrust::sort_by_key(thrust::cuda::par.on(stream),
                                    keys_device.begin(), keys_device.end(),
                                    param_ids_device.begin());

                m_stream.synchronize();
            }

            /*****************************************************************
             * Kernel5: Propagate to the next surface
             *****************************************************************/

            {
                // Reset the number of tracks per seed
                m_copy.memset(n_tracks_per_seed_buffer, 0)->ignore();

                const unsigned int nThreads = m_warp_size * 2;
                const unsigned int nBlocks =
                    (n_candidates + nThreads - 1) / nThreads;
                kernels::propagate_to_next_surface<propagator_type, bfield_type,
                                                   config_type>
                    <<<nBlocks, nThreads, 0, stream>>>(
                        m_cfg, det_view, field_view, in_params_buffer,
                        param_liveness_buffer, param_ids_buffer, link_map[step],
                        step, n_candidates, tips_buffer,
                        n_tracks_per_seed_buffer);
                TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

                m_stream.synchronize();
            }
        }

        // Fill the candidate size vector
        n_candidates_per_step.push_back(n_candidates);

        n_in_params = n_candidates;
    }

    // Create link buffer
    vecmem::data::jagged_vector_buffer<candidate_link> links_buffer(
        n_candidates_per_step, m_mr.main, m_mr.host);
    m_copy.setup(links_buffer)->ignore();

    // Copy link map to link buffer
    const auto n_steps = n_candidates_per_step.size();
    for (unsigned int it = 0; it < n_steps; it++) {

        vecmem::device_vector<candidate_link> in(link_map[it]);
        vecmem::device_vector<candidate_link> out(
            *(links_buffer.host_ptr() + it));

        thrust::copy(thrust::cuda::par.on(stream), in.begin(),
                     in.begin() + n_candidates_per_step[it], out.begin());
    }

    /*****************************************************************
     * Kernel6: Build tracks
     *****************************************************************/

    // Get the number of tips
    auto n_tips_total = m_copy.get_size(tips_buffer);

    // Create track candidate buffer
    track_candidate_container_types::buffer track_candidates_buffer{
        {n_tips_total, m_mr.main},
        {std::vector<std::size_t>(n_tips_total,
                                  m_cfg.max_track_candidates_per_track),
         m_mr.main, m_mr.host, vecmem::data::buffer_type::resizable}};

    m_copy.setup(track_candidates_buffer.headers)->ignore();
    m_copy.setup(track_candidates_buffer.items)->ignore();

    // Create buffer for valid indices
    vecmem::data::vector_buffer<unsigned int> valid_indices_buffer(n_tips_total,
                                                                   m_mr.main);

    unsigned int n_valid_tracks;

    // @Note: nBlocks can be zero in case there is no tip. This happens when
    // chi2_max config is set tightly and no tips are found
    if (n_tips_total > 0) {
        vecmem::unique_alloc_ptr<unsigned int> n_valid_tracks_device =
            vecmem::make_unique_alloc<unsigned int>(m_mr.main);
        TRACCC_CUDA_ERROR_CHECK(cudaMemsetAsync(n_valid_tracks_device.get(), 0,
                                                sizeof(unsigned int), stream));

        const unsigned int nThreads = m_warp_size * 2;
        const unsigned int nBlocks = (n_tips_total + nThreads - 1) / nThreads;

        kernels::build_tracks<<<nBlocks, nThreads, 0, stream>>>(
            m_cfg, measurements, seeds_buffer, links_buffer, tips_buffer,
            track_candidates_buffer, valid_indices_buffer,
            n_valid_tracks_device.get());
        TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

        // Global counter object: Device -> Host
        TRACCC_CUDA_ERROR_CHECK(cudaMemcpyAsync(
            &n_valid_tracks, n_valid_tracks_device.get(), sizeof(unsigned int),
            cudaMemcpyDeviceToHost, stream));

        m_stream.synchronize();
    }

    // Create pruned candidate buffer
    track_candidate_container_types::buffer prune_candidates_buffer{
        {n_valid_tracks, m_mr.main},
        {std::vector<std::size_t>(n_valid_tracks,
                                  m_cfg.max_track_candidates_per_track),
         m_mr.main, m_mr.host, vecmem::data::buffer_type::resizable}};

    m_copy.setup(prune_candidates_buffer.headers)->ignore();
    m_copy.setup(prune_candidates_buffer.items)->ignore();

    if (n_valid_tracks > 0) {
        const unsigned int nThreads = m_warp_size * 2;
        const unsigned int nBlocks = (n_valid_tracks + nThreads - 1) / nThreads;

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
