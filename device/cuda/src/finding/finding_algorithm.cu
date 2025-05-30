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
#include "./kernels/apply_interaction.hpp"
#include "./kernels/build_tracks.cuh"
#include "./kernels/fill_sort_keys.cuh"
#include "./kernels/find_tracks.cuh"
#include "./kernels/make_barcode_sequence.cuh"
#include "./kernels/propagate_to_next_surface.hpp"
#include "traccc/cuda/finding/finding_algorithm.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/find_nbest.hpp"
#include "traccc/edm/device/sort_key.hpp"
#include "traccc/finding/candidate_link.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/utils/detector_type_utils.hpp"
#include "traccc/utils/projections.hpp"
#include "traccc/utils/propagation.hpp"

// VecMem include(s).
#include <iterator>
#include <limits>
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

__global__ void gather_best_tips_per_measurement(
    const vecmem::data::vector_view<const unsigned int> tips_view,
    const vecmem::data::vector_view<const candidate_link> links_view,
    const measurement_collection_types::const_view measurements_view,
    vecmem::data::vector_view<unsigned long long int> insertion_mutex_view,
    vecmem::data::vector_view<unsigned int> tip_index_view,
    vecmem::data::vector_view<float> tip_pval_view,
    const unsigned int max_num_tracks_per_measurement) {
    unsigned int tip_idx = blockIdx.x * blockDim.x + threadIdx.x;

    const vecmem::device_vector<const unsigned int> tips(tips_view);
    const vecmem::device_vector<const candidate_link> links(links_view);
    const measurement_collection_types::const_device measurements(
        measurements_view);
    vecmem::device_vector<unsigned long long int> insertion_mutex(
        insertion_mutex_view);
    vecmem::device_vector<unsigned int> tip_index(tip_index_view);
    vecmem::device_vector<float> tip_pval(tip_pval_view);
    const unsigned int n_meas = measurements.size();

    scalar pval = 0.f;
    unsigned int link_idx = 0;
    unsigned int num_states = 0;

    bool need_to_write = true;
    candidate_link L;

    if (tip_idx < tips.size()) {
        link_idx = tips.at(tip_idx);
        const auto link = links.at(link_idx);
        pval = prob(link.chi2_sum, static_cast<scalar>(link.ndof_sum) - 5.f);
        num_states = link.step + 1 - link.n_skipped;

        L = link;

        // Skip any holes at the start; there shouldn't be any.
        while (L.meas_idx >= n_meas && L.step != 0u) {
            L = links.at(L.previous_candidate_idx);
        }
    } else {
        need_to_write = false;
    }

    unsigned int current_state = 0;

    while (__syncthreads_or(current_state < num_states || need_to_write)) {
        if (current_state < num_states || need_to_write) {
            assert(L.meas_idx < n_meas);

            if (need_to_write) {
                vecmem::device_atomic_ref<unsigned long long int> mutex(
                    insertion_mutex.at(L.meas_idx));

                unsigned long long int assumed = mutex.load();
                unsigned long long int desired_set;
                auto [locked, size, worst] =
                    device::decode_insertion_mutex(assumed);

                if (need_to_write && size >= max_num_tracks_per_measurement &&
                    pval <= worst) {
                    need_to_write = false;
                }

                bool holds_lock = false;

                if (need_to_write && !locked) {
                    desired_set =
                        device::encode_insertion_mutex(true, size, worst);

                    if (mutex.compare_exchange_strong(assumed, desired_set)) {
                        holds_lock = true;
                    }
                }

                if (holds_lock) {
                    unsigned int new_size;
                    unsigned int offset =
                        L.meas_idx * max_num_tracks_per_measurement;
                    unsigned int out_idx;

                    if (size == max_num_tracks_per_measurement) {
                        new_size = size;

                        scalar worst_pval = std::numeric_limits<scalar>::max();

                        for (unsigned int i = 0; i < size; ++i) {
                            if (tip_pval.at(offset + i) < worst_pval) {
                                worst_pval = tip_pval.at(offset + i);
                                out_idx = i;
                            }
                        }
                    } else {
                        new_size = size + 1;
                        out_idx = size;
                    }

                    tip_index.at(offset + out_idx) = tip_idx;
                    tip_pval.at(offset + out_idx) = pval;

                    scalar new_worst = std::numeric_limits<scalar>::max();

                    for (unsigned int i = 0; i < new_size; ++i) {
                        new_worst =
                            std::min(new_worst, tip_pval.at(offset + i));
                    }

                    [[maybe_unused]] bool cas_result =
                        mutex.compare_exchange_strong(
                            desired_set, device::encode_insertion_mutex(
                                             false, new_size, new_worst));

                    assert(cas_result);

                    need_to_write = false;
                }
            }

            if (!need_to_write) {
                if (current_state < num_states - 1) {
                    L = links.at(L.previous_candidate_idx);
                    while (L.meas_idx >= n_meas && L.step != 0u) {
                        L = links.at(L.previous_candidate_idx);
                    }
                    need_to_write = true;
                } else {
#ifndef NDEBUG
                    if (L.step != 0) {
                        do {
                            L = links.at(L.previous_candidate_idx);
                        } while (L.meas_idx >= n_meas && L.step != 0u);
                        assert(L.meas_idx >= n_meas);
                    }
                    assert(L.step == 0);
#endif
                }

                current_state++;
            }
        }
    }
}

__global__ void gather_measurement_votes(
    const vecmem::data::vector_view<const unsigned long long int>
        insertion_mutex_view,
    const vecmem::data::vector_view<const unsigned int> tip_index_view,
    vecmem::data::vector_view<unsigned int> votes_per_tip_view,
    const unsigned int max_num_tracks_per_measurement) {
    unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int measurement_idx = thread_idx / max_num_tracks_per_measurement;
    unsigned int tip_idx = thread_idx % max_num_tracks_per_measurement;

    const vecmem::device_vector<const unsigned long long int> insertion_mutex(
        insertion_mutex_view);
    const vecmem::device_vector<const unsigned int> tip_index(tip_index_view);
    vecmem::device_vector<unsigned int> votes_per_tip(votes_per_tip_view);

    if (measurement_idx >= insertion_mutex.size()) {
        return;
    }

    auto [locked, size, worst] =
        device::decode_insertion_mutex(insertion_mutex.at(measurement_idx));

    if (tip_idx < size) {
        vecmem::device_atomic_ref<unsigned int>(
            votes_per_tip.at(tip_index.at(thread_idx)))
            .fetch_add(1u);
    }
}

__global__ void update_tip_length_buffer(
    const vecmem::data::vector_view<const unsigned int> old_tip_length_view,
    vecmem::data::vector_view<unsigned int> new_tip_length_view,
    const vecmem::data::vector_view<const unsigned int> measurement_votes_view,
    unsigned int* tip_to_output_map, unsigned int* tip_to_output_map_idx,
    float min_measurement_voting_fraction) {
    const unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    assert(tip_to_output_map != nullptr);
    assert(tip_to_output_map_idx != nullptr);

    const vecmem::device_vector<const unsigned int> old_tip_length(
        old_tip_length_view);
    vecmem::device_vector<unsigned int> new_tip_length(new_tip_length_view);
    const vecmem::device_vector<const unsigned int> measurement_votes(
        measurement_votes_view);

    if (thread_idx >= measurement_votes_view.size()) {
        return;
    }

    const unsigned int total_measurements = old_tip_length.at(thread_idx);
    const unsigned int total_votes = measurement_votes.at(thread_idx);

    assert(total_votes <= total_measurements);

    const scalar vote_fraction = static_cast<scalar>(total_votes) /
                                 static_cast<scalar>(total_measurements);

    if (vote_fraction < min_measurement_voting_fraction) {
        tip_to_output_map[thread_idx] =
            std::numeric_limits<unsigned int>::max();
    } else {
        const auto new_idx =
            vecmem::device_atomic_ref(*tip_to_output_map_idx).fetch_add(1u);
        new_tip_length.at(new_idx) = total_measurements;
        tip_to_output_map[thread_idx] = new_idx;
    }
}

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
    const measurement_collection_types::const_view& measurements,
    const bound_track_parameters_collection_types::const_view& seeds_view)
    const {

    assert(m_cfg.min_step_length_for_next_surface >
               math::fabs(m_cfg.propagation.navigation.overstep_tolerance) &&
           "Min step length for the next surface should be higher than the "
           "overstep tolerance");

    // Get a convenience variable for the stream that we'll be using.
    cudaStream_t stream = details::get_stream(m_stream);

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

    const unsigned int n_seeds = m_copy.get_size(seeds_view);

    // Prepare input parameters with seeds
    bound_track_parameters_collection_types::buffer in_params_buffer(n_seeds,
                                                                     m_mr.main);
    m_copy.setup(in_params_buffer)->ignore();
    m_copy(seeds_view, in_params_buffer)->ignore();
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
    vecmem::data::vector_buffer<unsigned int> tip_length_buffer{
        m_cfg.max_num_branches_per_seed * n_seeds, m_mr.main};
    m_copy.setup(tip_length_buffer)->wait();

    std::map<unsigned int, unsigned int> step_to_link_idx_map;
    step_to_link_idx_map[0] = 0;

    unsigned int n_in_params = n_seeds;

    for (unsigned int step = 0; n_in_params > 0; step++) {

        /*****************************************************************
         * Kernel2: Apply material interaction
         ****************************************************************/

        {
            const unsigned int nThreads = m_warp_size * 2;
            const unsigned int nBlocks =
                (n_in_params + nThreads - 1) / nThreads;

            apply_interaction<std::decay_t<detector_type>>(
                nBlocks, nThreads, 0, stream, m_cfg,
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

            // Reset the number of tracks per seed
            m_copy.memset(n_tracks_per_seed_buffer, 0)->ignore();

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

            {
                vecmem::data::vector_buffer<candidate_link> tmp_links_buffer(
                    n_max_candidates, m_mr.main);
                m_copy.setup(tmp_links_buffer)->ignore();
                bound_track_parameters_collection_types::buffer
                    tmp_params_buffer(n_max_candidates, m_mr.main);
                m_copy.setup(tmp_params_buffer)->ignore();

                const unsigned int nThreads = m_warp_size * 2;
                const unsigned int nBlocks =
                    (n_in_params + nThreads - 1) / nThreads;

                const unsigned int prev_link_idx =
                    step == 0 ? 0 : step_to_link_idx_map[step - 1];

                assert(links_size == step_to_link_idx_map[step]);

                const std::size_t shared_size =
                    nThreads * sizeof(unsigned int) +
                    2 * nThreads *
                        sizeof(std::pair<unsigned int, unsigned int>);

                find_tracks<std::decay_t<detector_type>>(
                    nBlocks, nThreads, shared_size, stream, m_cfg,
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
                        .out_params_liveness_view = updated_liveness_buffer,
                        .tips_view = tips_buffer,
                        .tip_lengths_view = tip_length_buffer,
                        .n_tracks_per_seed_view = n_tracks_per_seed_buffer,
                        .tmp_params_view = tmp_params_buffer,
                        .tmp_links_view = tmp_links_buffer});
                TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

                std::swap(in_params_buffer, updated_params_buffer);
                std::swap(param_liveness_buffer, updated_liveness_buffer);

                m_stream.synchronize();

                step_to_link_idx_map[step + 1] = m_copy.get_size(links_buffer);
                n_candidates =
                    step_to_link_idx_map[step + 1] - step_to_link_idx_map[step];

                m_stream.synchronize();
            }
        }

        // If no more CKF step is expected, the tips and links are populated,
        // and any further time-consuming action is avoided
        if (step == m_cfg.max_track_candidates_per_track - 1) {
            break;
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
                const unsigned int nThreads = m_warp_size * 2;
                const unsigned int nBlocks =
                    (n_candidates + nThreads - 1) / nThreads;
                propagate_to_next_surface<std::decay_t<propagator_type>,
                                          std::decay_t<bfield_type>>(
                    nBlocks, nThreads, 0, stream, m_cfg,
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
                        .tip_lengths_view = tip_length_buffer});
                TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

                m_stream.synchronize();
            }
        }

        n_in_params = n_candidates;
    }

    TRACCC_DEBUG(
        "Final link buffer usage was "
        << m_copy.get_size(links_buffer) << " out of " << link_buffer_capacity
        << " ("
        << ((100.f * static_cast<float>(m_copy.get_size(links_buffer))) /
            static_cast<float>(link_buffer_capacity))
        << "%)");

    /*****************************************************************
     * Kernel6: Build tracks
     *****************************************************************/

    // Get the number of tips
    unsigned int n_tips_total = m_copy.get_size(tips_buffer);

    std::vector<unsigned int> tips_length_host;
    vecmem::unique_alloc_ptr<unsigned int[]> tip_to_output_map = nullptr;

    TRACCC_INFO("Before pruning we have " << n_tips_total << " tips");

    unsigned int n_tips_total_filtered = n_tips_total;

    if (n_tips_total > 0 && m_cfg.max_num_tracks_per_measurement > 0) {
        // TODO: DOCS

        vecmem::data::vector_buffer<unsigned int>
            best_tips_per_measurement_index_buffer(
                m_cfg.max_num_tracks_per_measurement * n_measurements,
                m_mr.main);
        m_copy.setup(best_tips_per_measurement_index_buffer)->wait();

        vecmem::data::vector_buffer<unsigned long long int>
            best_tips_per_measurement_insertion_mutex_buffer(n_measurements,
                                                             m_mr.main);
        m_copy.setup(best_tips_per_measurement_insertion_mutex_buffer)->wait();

        // NOTE: This memset assumes that an all-zero bit vector interpreted
        // as a floating point value has value zero, which is true for IEEE
        // 754 but might not be true for arbitrary float formats.
        m_copy.memset(best_tips_per_measurement_insertion_mutex_buffer, 0)
            ->wait();

        {
            vecmem::data::vector_buffer<float>
                best_tips_per_measurement_pval_buffer(
                    m_cfg.max_num_tracks_per_measurement * n_measurements,
                    m_mr.main);
            m_copy.setup(best_tips_per_measurement_pval_buffer)->wait();

            // NOTE: Normally, launching small blocks is a performance
            // antipattern, but there is little use to having larger blocks
            // here.
            const unsigned int num_threads = 32;
            const unsigned int num_blocks =
                (n_tips_total + num_threads - 1) / num_threads;

            gather_best_tips_per_measurement<<<num_blocks, num_threads, 0,
                                               stream>>>(
                tips_buffer, links_buffer, measurements,
                best_tips_per_measurement_insertion_mutex_buffer,
                best_tips_per_measurement_index_buffer,
                best_tips_per_measurement_pval_buffer,
                m_cfg.max_num_tracks_per_measurement);

            TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
        }

        vecmem::data::vector_buffer<unsigned int> votes_per_tip_buffer(
            n_tips_total, m_mr.main);
        m_copy.setup(votes_per_tip_buffer)->wait();
        m_copy.memset(votes_per_tip_buffer, 0)->wait();

        {
            const unsigned int num_threads = 512;
            const unsigned int num_blocks =
                (m_cfg.max_num_tracks_per_measurement * n_measurements +
                 num_threads - 1) /
                num_threads;

            gather_measurement_votes<<<num_blocks, num_threads, 0, stream>>>(
                best_tips_per_measurement_insertion_mutex_buffer,
                best_tips_per_measurement_index_buffer, votes_per_tip_buffer,
                m_cfg.max_num_tracks_per_measurement);

            TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
        }

        tip_to_output_map =
            vecmem::make_unique_alloc<unsigned int[]>(m_mr.main, n_tips_total);

        {
            const unsigned int num_threads = 512;
            const unsigned int num_blocks =
                (n_tips_total + num_threads - 1) / num_threads;

            vecmem::data::vector_buffer<unsigned int> new_tip_length_buffer{
                n_tips_total, m_mr.main};
            m_copy.setup(new_tip_length_buffer)->wait();

            auto tip_to_output_map_idx =
                vecmem::make_unique_alloc<unsigned int>(m_mr.main);

            TRACCC_CUDA_ERROR_CHECK(cudaMemsetAsync(
                tip_to_output_map_idx.get(), 0, sizeof(unsigned int), stream));

            update_tip_length_buffer<<<num_blocks, num_threads, 0, stream>>>(
                tip_length_buffer, new_tip_length_buffer, votes_per_tip_buffer,
                tip_to_output_map.get(), tip_to_output_map_idx.get(),
                m_cfg.min_measurement_voting_fraction);

            TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

            m_stream.synchronize();

            TRACCC_CUDA_ERROR_CHECK(cudaMemcpyAsync(
                &n_tips_total_filtered, tip_to_output_map_idx.get(),
                sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));

            tip_length_buffer = std::move(new_tip_length_buffer);

            m_stream.synchronize();
        }
    }

    TRACCC_INFO("After pruning we have " << n_tips_total_filtered << " tips");

    m_copy(tip_length_buffer, tips_length_host)->wait();
    tips_length_host.resize(n_tips_total_filtered);

    // Create track candidate buffer
    track_candidate_container_types::buffer track_candidates_buffer{
        {n_tips_total_filtered, m_mr.main},
        {tips_length_host, m_mr.main, m_mr.host}};

    m_copy.setup(track_candidates_buffer.headers)->ignore();
    m_copy.setup(track_candidates_buffer.items)->ignore();

    // @Note: nBlocks can be zero in case there is no tip. This happens when
    // chi2_max config is set tightly and no tips are found
    if (n_tips_total > 0) {
        const unsigned int nThreads = m_warp_size * 2;
        const unsigned int nBlocks = (n_tips_total + nThreads - 1) / nThreads;

        kernels::build_tracks<<<nBlocks, nThreads, 0, stream>>>(
            device::build_tracks_payload{
                .measurements_view = measurements,
                .seeds_view = seeds_view,
                .links_view = links_buffer,
                .tips_view = tips_buffer,
                .track_candidates_view = track_candidates_buffer,
                .tip_to_output_map = tip_to_output_map.get()});
        TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

        m_stream.synchronize();
    }

    return track_candidates_buffer;
}

// Explicit template instantiation
template class finding_algorithm<
    stepper_for_t<::traccc::default_detector::device>,
    navigator_for_t<::traccc::default_detector::device>>;
}  // namespace traccc::cuda
