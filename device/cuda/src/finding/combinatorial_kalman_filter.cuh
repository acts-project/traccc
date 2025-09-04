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
#include "../utils/get_size.hpp"
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
    const typename detector_t::const_view_type& det, const bfield_t& field,
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

    vecmem::unique_alloc_ptr<unsigned int> size_staging_ptr =
        vecmem::make_unique_alloc<unsigned int>(*(mr.host));

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

        const unsigned int links_size = step_to_link_idx_map[step];

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

            TRACCC_CUDA_ERROR_CHECK(cudaMemcpyAsync(
                size_staging_ptr.get(), links_buffer.size_ptr(),
                sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));

            str.synchronize();

            step_to_link_idx_map[step + 1] = *size_staging_ptr;
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
    auto n_tips_total = get_size(tips_buffer, size_staging_ptr.get(), stream);

    vecmem::vector<unsigned int> tips_length_host(mr.host);

    if (n_tips_total > 0) {
        copy(tip_length_buffer, tips_length_host)->wait();
        tips_length_host.resize(n_tips_total);
    }

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
             .track_candidates_view = {track_candidates_buffer, measurements}});
        TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

        str.synchronize();
    }

    return track_candidates_buffer;
}

}  // namespace traccc::cuda::details
