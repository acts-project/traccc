/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "../sanity/contiguous_on.cuh"
#include "../utils/barrier.hpp"
#include "../utils/cuda_error_handling.hpp"
#include "../utils/thread_id.hpp"
#include "../utils/utils.hpp"
#include "./kernels/apply_interaction.cuh"
#include "./kernels/build_tracks.cuh"
#include "./kernels/fill_sort_keys.cuh"
#include "./kernels/find_tracks.cuh"
#include "./kernels/make_barcode_sequence.cuh"
#include "./kernels/propagate_to_next_surface.cuh"
#include "./kernels/propagate_stage1.cuh"
#include "./kernels/propagate_stage2.cuh"
#include "./kernels/prune_tracks.cuh"
#include "traccc/cuda/finding/finding_algorithm.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/device/sort_key.hpp"
#include "traccc/finding/candidate_link.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/utils/detector_type_utils.hpp"
#include "traccc/utils/projections.hpp"
#include "traccc/utils/propagation.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_buffer.hpp>
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/jagged_device_vector.hpp>
#include <vecmem/containers/vector.hpp>
#include <vecmem/memory/unique_ptr.hpp>

// Thrust include(s).
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

// System include(s).
#include <cassert>
#include <memory_resource>
#include <vector>

namespace traccc::cuda {

template <typename stepper_t, typename navigator_t>
finding_algorithm<stepper_t, navigator_t>::finding_algorithm(
    const config_type& cfg, const traccc::memory_resource& mr,
    vecmem::copy& copy, stream& str, std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)),
      m_cfg(cfg),
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

    assert(m_cfg.min_step_length_for_next_surface >
               math::fabs(m_cfg.propagation.navigation.overstep_tolerance) &&
           "Min step length for the next surface should be higher than the "
           "overstep tolerance");

    // Get a convenience variable for the stream that we'll be using.
    cudaStream_t stream = details::get_stream(m_stream);

    // Copy setup
    m_copy.setup(seeds_buffer)->ignore();

    // The Thrust policy to use.
    auto thrust_policy =
        thrust::cuda::par_nosync(std::pmr::polymorphic_allocator(&(m_mr.main)))
            .on(stream);

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

        measurement* uniques_end =
            thrust::unique_copy(thrust_policy, measurements.ptr(),
                                measurements.ptr() + n_measurements,
                                uniques.begin(), measurement_equal_comp());
        m_stream.synchronize();
        n_modules = static_cast<unsigned int>(uniques_end - uniques.begin());
    }

    // Get upper bounds of unique elements
    vecmem::data::vector_buffer<unsigned int> upper_bounds_buffer{n_modules,
                                                                  m_mr.main};
    m_copy.setup(upper_bounds_buffer)->ignore();

    {
        vecmem::device_vector<unsigned int> upper_bounds(upper_bounds_buffer);

        measurement_collection_types::device uniques(uniques_buffer);

        thrust::upper_bound(thrust_policy, measurements.ptr(),
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
            device::make_barcode_sequence_payload{
                .uniques_view = uniques_buffer,
                .barcodes_view = barcodes_buffer});

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

    // Create a buffer for links
    unsigned int link_buffer_capacity = m_cfg.initial_links_per_seed * n_seeds;
    vecmem::data::vector_buffer<candidate_link> links_buffer(
        link_buffer_capacity, m_mr.main, vecmem::data::buffer_type::resizable);
    m_copy.setup(links_buffer)->wait();

    // Create a buffer of tip links
    vecmem::data::vector_buffer<unsigned int> tips_buffer{
        m_cfg.max_num_branches_per_seed * n_seeds, m_mr.main,
        vecmem::data::buffer_type::resizable};
    m_copy.setup(tips_buffer)->wait();

    std::map<unsigned int, unsigned int> step_to_link_idx_map;
    step_to_link_idx_map[0] = 0;

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

            kernels::apply_interaction<
                std::decay_t<detector_type>><<<nBlocks, nThreads, 0, stream>>>(
                m_cfg,
                device::apply_interaction_payload<std::decay_t<detector_type>>{
                    .det_data = det_view,
                    .n_params = n_in_params,
                    .params_view = in_params_buffer,
                    .params_liveness_view = param_liveness_buffer});
            TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
        }

        /*****************************************************************
         * Kernel3: Find valid tracks
         *****************************************************************/

        unsigned int n_candidates = 0;

        {
            // Buffer for kalman-updated parameters spawned by the measurement
            // candidates
            const unsigned int n_max_candidates =
                n_in_params * m_cfg.max_num_branches_per_surface;

            bound_track_parameters_collection_types::buffer
                updated_params_buffer(n_max_candidates, m_mr.main);
            m_copy.setup(updated_params_buffer)->ignore();

            vecmem::data::vector_buffer<unsigned int> updated_liveness_buffer(
                n_max_candidates, m_mr.main);
            m_copy.setup(updated_liveness_buffer)->ignore();

            const unsigned int links_size = m_copy.get_size(links_buffer);

            if (links_size + n_max_candidates > link_buffer_capacity) {
                const unsigned int new_link_buffer_capacity = std::max(
                    2 * link_buffer_capacity, links_size + n_max_candidates);

                TRACCC_INFO("Link buffer (capacity "
                            << link_buffer_capacity << ") is too small to hold "
                            << links_size << " current and " << n_max_candidates
                            << " new links; increasing capacity to "
                            << new_link_buffer_capacity);

                link_buffer_capacity = new_link_buffer_capacity;

                vecmem::data::vector_buffer<candidate_link> new_links_buffer(
                    link_buffer_capacity, m_mr.main,
                    vecmem::data::buffer_type::resizable);

                m_copy.setup(new_links_buffer)->wait();
                m_copy(links_buffer, new_links_buffer)->wait();

                links_buffer = std::move(new_links_buffer);
            }

            const unsigned int nThreads = m_warp_size * 2;
            const unsigned int nBlocks =
                (n_in_params + nThreads - 1) / nThreads;

            const unsigned int prev_link_idx =
                step == 0 ? 0 : step_to_link_idx_map[step - 1];

            assert(links_size == step_to_link_idx_map[step]);

            kernels::find_tracks<std::decay_t<detector_type>>
                <<<nBlocks, nThreads,
                   nThreads * sizeof(unsigned int) +
                       2 * nThreads *
                           sizeof(std::pair<unsigned int, unsigned int>),
                   stream>>>(
                    m_cfg,
                    device::find_tracks_payload<std::decay_t<detector_type>>{
                        .det_data = det_view,
                        .measurements_view = measurements,
                        .in_params_view = in_params_buffer,
                        .in_params_liveness_view = param_liveness_buffer,
                        .n_in_params = n_in_params,
                        .barcodes_view = barcodes_buffer,
                        .upper_bounds_view = upper_bounds_buffer,
                        .links_view = links_buffer,
                        .prev_links_idx = prev_link_idx,
                        .curr_links_idx = step_to_link_idx_map[step],
                        .step = step,
                        .out_params_view = updated_params_buffer,
                        .out_params_liveness_view = updated_liveness_buffer});
            TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

            std::swap(in_params_buffer, updated_params_buffer);
            std::swap(param_liveness_buffer, updated_liveness_buffer);

            m_stream.synchronize();

            step_to_link_idx_map[step + 1] = m_copy.get_size(links_buffer);
            n_candidates =
                step_to_link_idx_map[step + 1] - step_to_link_idx_map[step];

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
                    device::fill_sort_keys_payload{
                        .params_view = in_params_buffer,
                        .keys_view = keys_buffer,
                        .ids_view = param_ids_buffer});
                TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

                // Sort the key and values
                vecmem::device_vector<device::sort_key> keys_device(
                    keys_buffer);
                vecmem::device_vector<unsigned int> param_ids_device(
                    param_ids_buffer);
                thrust::sort_by_key(thrust_policy, keys_device.begin(),
                                    keys_device.end(),
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
                /* ---------- Stage-1：粗步進 ---------- */
                kernels::propagate_stage1<
                    std::decay_t<propagator_type>, std::decay_t<bfield_type>>
                    <<<nBlocks, nThreads, 0, stream>>>(
                        m_cfg,
                        device::propagate_to_next_surface_payload<
                            std::decay_t<propagator_type>,
                            std::decay_t<bfield_type>>{
                            .det_data = det_view,
                            .field_data = field_view,
                            .params_view = in_params_buffer,
                            .params_liveness_view = param_liveness_buffer,
                            .param_ids_view = param_ids_buffer,
                            .links_view = links_buffer,
                            .prev_links_idx = step_to_link_idx_map[step],
                            .step = step,
                            .n_in_params = n_candidates,
                            .tips_view = tips_buffer,
                            .n_tracks_per_seed_view =
                                n_tracks_per_seed_buffer});
                TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

                m_stream.synchronize();

                /* ---------- Stage-2：Covariance 更新 ---------- */
                kernels::propagate_stage2<
                    default_finding_algorithm::propagator_type,
                    default_finding_algorithm::bfield_type>
                    <<<nBlocks, nThreads, 0, m_stream>>>(
                        m_cfg,
                        typename device::propagate_to_next_surface_payload<
                            default_finding_algorithm::propagator_type,
                            default_finding_algorithm::bfield_type>{
                            .det_data               = det_view,
                            .field_data             = field_view,
                            .params_view            = in_params_buffer,
                            .params_liveness_view   = param_liveness_buffer,
                            .param_ids_view         = param_ids_buffer,
                            .links_view             = links_buffer,
                            .prev_links_idx         = step_to_link_idx_map[step],
                            .step                   = step,
                            .n_in_params            = n_candidates,
                            .tips_view              = tips_buffer,
                            .n_tracks_per_seed_view = n_tracks_per_seed_buffer});
                TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
            }
        }

        n_in_params = n_candidates;
    }

    TRACCC_DEBUG(
        "Final link buffer usage was "
        << m_copy.get_size(links_buffer) << " out of " << link_buffer_capacity
        << " ("
        << ((100.f * m_copy.get_size(links_buffer)) / link_buffer_capacity)
        << "%)");

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

    unsigned int n_valid_tracks = 0;

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
            m_cfg, device::build_tracks_payload{
                       .measurements_view = measurements,
                       .seeds_view = seeds_buffer,
                       .links_view = links_buffer,
                       .tips_view = tips_buffer,
                       .track_candidates_view = track_candidates_buffer,
                       .valid_indices_view = valid_indices_buffer,
                       .n_valid_tracks = n_valid_tracks_device.get()});
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
            device::prune_tracks_payload{
                .track_candidates_view = track_candidates_buffer,
                .valid_indices_view = valid_indices_buffer,
                .prune_candidates_view = prune_candidates_buffer});
        TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
    }

    return prune_candidates_buffer;
}

// Explicit template instantiation
template class finding_algorithm<
    stepper_for_t<::traccc::default_detector::device>,
    navigator_for_t<::traccc::default_detector::device>>;
}  // namespace traccc::cuda
