/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "../sanity/contiguous_on.cuh"
#include "../utils/barrier.hpp"
#include "../utils/cuda_error_handling.hpp"
#include "../utils/thread_id.hpp"
#include "../utils/utils.hpp"
#include "./kernels/apply_interaction.hpp"
#include "./kernels/build_tracks.cuh"
#include "./kernels/fill_finding_duplicate_removal_sort_keys.cuh"
#include "./kernels/fill_finding_propagation_sort_keys.cuh"
#include "./kernels/find_tracks.cuh"
#include "./kernels/make_barcode_sequence.cuh"
#include "./kernels/propagate_to_next_surface.hpp"
#include "./kernels/remove_duplicates.cuh"

// Project include(s).
#include "traccc/device/find_nbest.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_candidate_collection.hpp"
#include "traccc/finding/candidate_link.hpp"
#include "traccc/finding/details/combinatorial_kalman_filter_types.hpp"
#include "traccc/finding/finding_config.hpp"
#include "traccc/utils/logging.hpp"
#include "traccc/utils/memory_resource.hpp"
#include "traccc/utils/projections.hpp"
#include "traccc/utils/propagation.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

// Thrust include(s).
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <iterator>
#include <limits>

namespace traccc::cuda {

__global__ inline void gather_best_tips_per_measurement(
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
        pval = prob(link.chi2_sum, static_cast<scalar>(link.ndf_sum) - 5.f);
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

__global__ inline void gather_measurement_votes(
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

__global__ inline void update_tip_length_buffer(
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
}  // namespace traccc::cuda

namespace traccc::cuda::details {

/// Templated implementation of the track finding algorithm.
///
/// Concrete track finding algorithms can use this function with the appropriate
/// specializations, to find tracks on top of a specific detector type, magnetic
/// field type, and track finding configuration.
///
/// @tparam detector_t The (device) detector type to use
/// @tparam bfield_t   The magnetic field type to use
///
/// @param det               A view of the detector object
/// @param field             The magnetic field object
/// @param measurements_view All measurements in an event
/// @param seeds_view        All seeds in an event to start the track finding
///                          with
/// @param config            The track finding configuration
/// @param mr                The memory resource(s) to use
/// @param copy              The copy object to use
/// @param log               The logger to use for message logging
/// @param stream            The CUDA stream to use for the operations
/// @param warp_size         The warp size of the used CUDA device
///
/// @return A buffer of the found track candidates
///
template <typename detector_t, typename bfield_t>
edm::track_candidate_collection<default_algebra>::buffer
combinatorial_kalman_filter(
    const typename detector_t::view_type& det, const bfield_t& field,
    const measurement_collection_types::const_view& measurements,
    const bound_track_parameters_collection_types::const_view& seeds,
    const finding_config& config, const memory_resource& mr, vecmem::copy& copy,
    const Logger& log, stream& str, unsigned int warp_size) {

    assert(config.min_step_length_for_next_surface >
               math::fabs(config.propagation.navigation.overstep_tolerance) &&
           "Min step length for the next surface should be higher than the "
           "overstep tolerance");
    assert(is_contiguous_on<measurement_collection_types::const_device>(
        measurement_module_projection(), mr.main, copy, str, measurements));

    // Create a logger.
    auto logger = [&log]() -> const Logger& { return log; };

    /// Access the underlying CUDA stream.
    cudaStream_t stream = get_stream(str);

    /// Thrust policy to use.
    auto thrust_policy =
        thrust::cuda::par_nosync(std::pmr::polymorphic_allocator(&(mr.main)))
            .on(stream);

    /*****************************************************************
     * Measurement Operations
     *****************************************************************/

    const measurement_collection_types::const_view::size_type n_measurements =
        copy.get_size(measurements);

    // Get copy of barcode uniques
    measurement_collection_types::buffer uniques_buffer{n_measurements,
                                                        mr.main};
    copy.setup(uniques_buffer)->ignore();
    measurement_collection_types::device uniques(uniques_buffer);

    measurement_collection_types::device::iterator uniques_end =
        thrust::unique_copy(thrust_policy, measurements.ptr(),
                            measurements.ptr() + n_measurements,
                            uniques.begin(), measurement_equal_comp());
    str.synchronize();
    const unsigned int n_modules =
        static_cast<unsigned int>(uniques_end - uniques.begin());

    // Get upper bounds of unique elements
    vecmem::data::vector_buffer<unsigned int> upper_bounds_buffer{n_modules,
                                                                  mr.main};
    copy.setup(upper_bounds_buffer)->ignore();
    vecmem::device_vector<unsigned int> upper_bounds(upper_bounds_buffer);

    thrust::upper_bound(thrust_policy, measurements.ptr(),
                        measurements.ptr() + n_measurements, uniques.begin(),
                        uniques.begin() + n_modules, upper_bounds.begin(),
                        measurement_sort_comp());

    /*****************************************************************
     * Kernel1: Create barcode sequence
     *****************************************************************/

    vecmem::data::vector_buffer<detray::geometry::barcode> barcodes_buffer{
        n_modules, mr.main};
    copy.setup(barcodes_buffer)->ignore();

    {
        const unsigned int nThreads = warp_size * 2;
        const unsigned int nBlocks =
            (barcodes_buffer.size() + nThreads - 1) / nThreads;

        kernels::make_barcode_sequence<<<nBlocks, nThreads, 0, stream>>>(
            device::make_barcode_sequence_payload{
                .uniques_view = uniques_buffer,
                .barcodes_view = barcodes_buffer});

        TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
    }

    const unsigned int n_seeds = copy.get_size(seeds);

    // Prepare input parameters with seeds
    bound_track_parameters_collection_types::buffer in_params_buffer(n_seeds,
                                                                     mr.main);
    copy.setup(in_params_buffer)->ignore();
    copy(seeds, in_params_buffer, vecmem::copy::type::device_to_device)
        ->ignore();
    vecmem::data::vector_buffer<unsigned int> param_liveness_buffer(n_seeds,
                                                                    mr.main);
    copy.setup(param_liveness_buffer)->ignore();
    copy.memset(param_liveness_buffer, 1)->ignore();

    // Number of tracks per seed
    vecmem::data::vector_buffer<unsigned int> n_tracks_per_seed_buffer(n_seeds,
                                                                       mr.main);
    copy.setup(n_tracks_per_seed_buffer)->ignore();

    // Create a buffer for links
    unsigned int link_buffer_capacity = config.initial_links_per_seed * n_seeds;
    vecmem::data::vector_buffer<candidate_link> links_buffer(
        link_buffer_capacity, mr.main, vecmem::data::buffer_type::resizable);
    copy.setup(links_buffer)->ignore();

    // Create a buffer of tip links
    vecmem::data::vector_buffer<unsigned int> tips_buffer{
        config.max_num_branches_per_seed * n_seeds, mr.main,
        vecmem::data::buffer_type::resizable};
    copy.setup(tips_buffer)->ignore();
    vecmem::data::vector_buffer<unsigned int> tip_length_buffer{
        config.max_num_branches_per_seed * n_seeds, mr.main};
    copy.setup(tip_length_buffer)->ignore();

    std::map<unsigned int, unsigned int> step_to_link_idx_map;
    step_to_link_idx_map[0] = 0;

    unsigned int n_in_params = n_seeds;
    for (unsigned int step = 0;
         step < config.max_track_candidates_per_track && n_in_params > 0;
         step++) {

        /*****************************************************************
         * Kernel2: Apply material interaction
         ****************************************************************/

        {
            const unsigned int nThreads = warp_size * 2;
            const unsigned int nBlocks =
                (n_in_params + nThreads - 1) / nThreads;

            apply_interaction<detector_t>(
                nBlocks, nThreads, 0, stream, config,
                device::apply_interaction_payload<detector_t>{
                    .det_data = det,
                    .n_params = n_in_params,
                    .params_view = in_params_buffer,
                    .params_liveness_view = param_liveness_buffer});
            TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
        }

        /*****************************************************************
         * Kernel3: Find valid tracks
         *****************************************************************/

        unsigned int n_candidates = 0;

        // Buffer for kalman-updated parameters spawned by the
        // measurement candidates
        const unsigned int n_max_candidates =
            n_in_params * config.max_num_branches_per_surface;

        bound_track_parameters_collection_types::buffer updated_params_buffer(
            n_max_candidates, mr.main);
        copy.setup(updated_params_buffer)->ignore();

        vecmem::data::vector_buffer<unsigned int> updated_liveness_buffer(
            n_max_candidates, mr.main);
        copy.setup(updated_liveness_buffer)->ignore();

        // Reset the number of tracks per seed
        copy.memset(n_tracks_per_seed_buffer, 0)->ignore();

        const unsigned int links_size = copy.get_size(links_buffer);

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
                link_buffer_capacity, mr.main,
                vecmem::data::buffer_type::resizable);

            copy.setup(new_links_buffer)->ignore();
            copy(links_buffer, new_links_buffer)->wait();

            links_buffer = std::move(new_links_buffer);
        }

        {
            vecmem::data::vector_buffer<candidate_link> tmp_links_buffer(
                n_max_candidates, mr.main);
            copy.setup(tmp_links_buffer)->ignore();
            bound_track_parameters_collection_types::buffer tmp_params_buffer(
                n_max_candidates, mr.main);
            copy.setup(tmp_params_buffer)->ignore();

            // Allocate the kernel's payload in host memory.
            using payload_t = device::find_tracks_payload<detector_t>;
            const payload_t host_payload{
                .det_data = det,
                .measurements_view = measurements,
                .in_params_view = in_params_buffer,
                .in_params_liveness_view = param_liveness_buffer,
                .n_in_params = n_in_params,
                .barcodes_view = barcodes_buffer,
                .upper_bounds_view = upper_bounds_buffer,
                .links_view = links_buffer,
                .prev_links_idx =
                    (step == 0 ? 0 : step_to_link_idx_map[step - 1]),
                .curr_links_idx = step_to_link_idx_map[step],
                .step = step,
                .out_params_view = updated_params_buffer,
                .out_params_liveness_view = updated_liveness_buffer,
                .tips_view = tips_buffer,
                .tip_lengths_view = tip_length_buffer,
                .n_tracks_per_seed_view = n_tracks_per_seed_buffer,
                .tmp_params_view = tmp_params_buffer,
                .tmp_links_view = tmp_links_buffer};

            // The number of threads, blocks and shared memory to use.
            const unsigned int nThreads = warp_size * 2;
            const unsigned int nBlocks =
                (n_in_params + nThreads - 1) / nThreads;
            const std::size_t shared_size =
                nThreads * sizeof(unsigned long long int) +
                2 * nThreads * sizeof(std::pair<unsigned int, unsigned int>);

            // Run the kernel.
            find_tracks<detector_t>(nBlocks, nThreads, shared_size, stream,
                                    config, host_payload);
            TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

            std::swap(in_params_buffer, updated_params_buffer);
            std::swap(param_liveness_buffer, updated_liveness_buffer);

            str.synchronize();

            step_to_link_idx_map[step + 1] = copy.get_size(links_buffer);
            n_candidates =
                step_to_link_idx_map[step + 1] - step_to_link_idx_map[step];
        }

        /*
         * On later steps, we can duplicate removal which will attempt to find
         * tracks that are propagated multiple times and deduplicate them.
         */
        if (n_candidates > 0 &&
            step >= config.duplicate_removal_minimum_length) {
            vecmem::data::vector_buffer<unsigned int>
                link_last_measurement_buffer(n_candidates, mr.main);
            vecmem::data::vector_buffer<unsigned int> param_ids_buffer(
                n_candidates, mr.main);

            /*
             * First, we sort the tracks by the index of their final
             * measurement which is critical to ensure good performance.
             */
            {
                const unsigned int nThreads = 256;
                const unsigned int nBlocks =
                    (n_candidates + nThreads - 1) / nThreads;

                kernels::fill_finding_duplicate_removal_sort_keys<<<
                    nBlocks, nThreads, 0, stream>>>(
                    {.links_view = links_buffer,
                     .param_liveness_view = param_liveness_buffer,
                     .link_last_measurement_view = link_last_measurement_buffer,
                     .param_ids_view = param_ids_buffer,
                     .n_links = n_candidates,
                     .curr_links_idx = step_to_link_idx_map[step],
                     .n_measurements = n_measurements});

                TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
            }

            vecmem::device_vector<unsigned int> keys_device(
                link_last_measurement_buffer);
            vecmem::device_vector<unsigned int> param_ids_device(
                param_ids_buffer);
            thrust::sort_by_key(thrust_policy, keys_device.begin(),
                                keys_device.end(), param_ids_device.begin());

            /*
             * Then, we run the actual duplicate removal kernel.
             */
            {
                const unsigned int nThreads = 256;
                const unsigned int nBlocks =
                    (n_candidates + nThreads - 1) / nThreads;

                kernels::remove_duplicates<<<nBlocks, nThreads, 0, stream>>>(
                    config,
                    {.links_view = links_buffer,
                     .link_last_measurement_view = link_last_measurement_buffer,
                     .param_ids_view = param_ids_buffer,
                     .param_liveness_view = param_liveness_buffer,
                     .n_links = n_candidates,
                     .curr_links_idx = step_to_link_idx_map[step],
                     .n_measurements = n_measurements,
                     .step = step});
            }

            TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
        }

        // If no more CKF step is expected, the tips and links are populated,
        // and any further time-consuming action is avoided
        if (step == config.max_track_candidates_per_track - 1) {
            break;
        }

        if (n_candidates > 0) {
            /*****************************************************************
             * Kernel4: Get key and value for parameter sorting
             *****************************************************************/

            vecmem::data::vector_buffer<unsigned int> param_ids_buffer(
                n_candidates, mr.main);
            copy.setup(param_ids_buffer)->ignore();

            {
                vecmem::data::vector_buffer<device::sort_key> keys_buffer(
                    n_candidates, mr.main);
                copy.setup(keys_buffer)->wait();

                const unsigned int nThreads = warp_size * 2;
                const unsigned int nBlocks =
                    (n_candidates + nThreads - 1) / nThreads;
                kernels::fill_finding_propagation_sort_keys<<<nBlocks, nThreads,
                                                              0, stream>>>(
                    {.params_view = in_params_buffer,
                     .param_liveness_view = param_liveness_buffer,
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
                str.synchronize();
            }

            /*****************************************************************
             * Kernel5: Propagate to the next surface
             *****************************************************************/

            {
                // Allocate the kernel's payload in host memory.
                using payload_t = device::propagate_to_next_surface_payload<
                    traccc::details::ckf_propagator_t<detector_t, bfield_t>,
                    bfield_t>;
                const payload_t host_payload{
                    .det_data = det,
                    .field_data = field,
                    .params_view = in_params_buffer,
                    .params_liveness_view = param_liveness_buffer,
                    .param_ids_view = param_ids_buffer,
                    .links_view = links_buffer,
                    .prev_links_idx = step_to_link_idx_map[step],
                    .step = step,
                    .n_in_params = n_candidates,
                    .tips_view = tips_buffer,
                    .tip_lengths_view = tip_length_buffer};

                const unsigned int nThreads = warp_size * 4;
                const unsigned int nBlocks =
                    (n_candidates + nThreads - 1) / nThreads;
                propagate_to_next_surface<
                    traccc::details::ckf_propagator_t<detector_t, bfield_t>,
                    bfield_t>(nBlocks, nThreads, 0, stream, config,
                              host_payload);
                TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

                str.synchronize();
            }
        }

        n_in_params = n_candidates;
    }

    TRACCC_DEBUG("Final link buffer usage was "
                 << copy.get_size(links_buffer) << " out of "
                 << link_buffer_capacity << " ("
                 << ((100.f * static_cast<float>(copy.get_size(links_buffer))) /
                     static_cast<float>(link_buffer_capacity))
                 << "%)");

    /*****************************************************************
     * Kernel6: Build tracks
     *****************************************************************/

    // Get the number of tips
    unsigned int n_tips_total = copy.get_size(tips_buffer);

    std::vector<unsigned int> tips_length_host;
    vecmem::unique_alloc_ptr<unsigned int[]> tip_to_output_map = nullptr;

    TRACCC_INFO("Before pruning we have " << n_tips_total << " tips");

    unsigned int n_tips_total_filtered = n_tips_total;

    if (n_tips_total > 0 && config.max_num_tracks_per_measurement > 0) {
        // TODO: DOCS

        vecmem::data::vector_buffer<unsigned int>
            best_tips_per_measurement_index_buffer(
                config.max_num_tracks_per_measurement * n_measurements,
                mr.main);
        copy.setup(best_tips_per_measurement_index_buffer)->wait();

        vecmem::data::vector_buffer<unsigned long long int>
            best_tips_per_measurement_insertion_mutex_buffer(n_measurements,
                                                             mr.main);
        copy.setup(best_tips_per_measurement_insertion_mutex_buffer)->wait();

        // NOTE: This memset assumes that an all-zero bit vector interpreted
        // as a floating point value has value zero, which is true for IEEE
        // 754 but might not be true for arbitrary float formats.
        copy.memset(best_tips_per_measurement_insertion_mutex_buffer, 0)
            ->wait();

        {
            vecmem::data::vector_buffer<float>
                best_tips_per_measurement_pval_buffer(
                    config.max_num_tracks_per_measurement * n_measurements,
                    mr.main);
            copy.setup(best_tips_per_measurement_pval_buffer)->wait();

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
                config.max_num_tracks_per_measurement);

            TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
        }

        vecmem::data::vector_buffer<unsigned int> votes_per_tip_buffer(
            n_tips_total, mr.main);
        copy.setup(votes_per_tip_buffer)->wait();
        copy.memset(votes_per_tip_buffer, 0)->wait();

        {
            const unsigned int num_threads = 512;
            const unsigned int num_blocks =
                (config.max_num_tracks_per_measurement * n_measurements +
                 num_threads - 1) /
                num_threads;

            gather_measurement_votes<<<num_blocks, num_threads, 0, stream>>>(
                best_tips_per_measurement_insertion_mutex_buffer,
                best_tips_per_measurement_index_buffer, votes_per_tip_buffer,
                config.max_num_tracks_per_measurement);

            TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
        }

        tip_to_output_map =
            vecmem::make_unique_alloc<unsigned int[]>(mr.main, n_tips_total);

        {
            const unsigned int num_threads = 512;
            const unsigned int num_blocks =
                (n_tips_total + num_threads - 1) / num_threads;

            vecmem::data::vector_buffer<unsigned int> new_tip_length_buffer{
                n_tips_total, mr.main};
            copy.setup(new_tip_length_buffer)->wait();

            auto tip_to_output_map_idx =
                vecmem::make_unique_alloc<unsigned int>(mr.main);

            TRACCC_CUDA_ERROR_CHECK(cudaMemsetAsync(
                tip_to_output_map_idx.get(), 0, sizeof(unsigned int), stream));

            update_tip_length_buffer<<<num_blocks, num_threads, 0, stream>>>(
                tip_length_buffer, new_tip_length_buffer, votes_per_tip_buffer,
                tip_to_output_map.get(), tip_to_output_map_idx.get(),
                config.min_measurement_voting_fraction);

            TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

            str.synchronize();

            TRACCC_CUDA_ERROR_CHECK(cudaMemcpyAsync(
                &n_tips_total_filtered, tip_to_output_map_idx.get(),
                sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));

            tip_length_buffer = std::move(new_tip_length_buffer);

            str.synchronize();
        }
    }

    TRACCC_INFO("After pruning we have " << n_tips_total_filtered << " tips");

    copy(tip_length_buffer, tips_length_host)->wait();
    tips_length_host.resize(n_tips_total_filtered);

    // Create track candidate buffer
    edm::track_candidate_collection<default_algebra>::buffer
        track_candidates_buffer{tips_length_host, mr.main, mr.host};
    copy.setup(track_candidates_buffer)->ignore();

    // @Note: nBlocks can be zero in case there is no tip. This happens when
    // chi2_max config is set tightly and no tips are found
    if (n_tips_total > 0) {
        const unsigned int nThreads = warp_size * 2;
        const unsigned int nBlocks = (n_tips_total + nThreads - 1) / nThreads;

        kernels::build_tracks<<<nBlocks, nThreads, 0, stream>>>(
            {.seeds_view = seeds,
             .links_view = links_buffer,
             .tips_view = tips_buffer,
             .track_candidates_view = {track_candidates_buffer, measurements},
             .tip_to_output_map = tip_to_output_map.get()});
        TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

        str.synchronize();
    }

    return track_candidates_buffer;
}

}  // namespace traccc::cuda::details
