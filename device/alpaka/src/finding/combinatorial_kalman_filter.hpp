/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "../utils/barrier.hpp"
#include "../utils/parallel_algorithms.hpp"
#include "../utils/thread_id.hpp"
#include "../utils/utils.hpp"

// Project include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_candidate_collection.hpp"
#include "traccc/finding/actors/ckf_aborter.hpp"
#include "traccc/finding/actors/interaction_register.hpp"
#include "traccc/finding/candidate_link.hpp"
#include "traccc/finding/details/combinatorial_kalman_filter_types.hpp"
#include "traccc/finding/device/apply_interaction.hpp"
#include "traccc/finding/device/build_tracks.hpp"
#include "traccc/finding/device/fill_finding_duplicate_removal_sort_keys.hpp"
#include "traccc/finding/device/fill_finding_propagation_sort_keys.hpp"
#include "traccc/finding/device/find_tracks.hpp"
#include "traccc/finding/device/make_barcode_sequence.hpp"
#include "traccc/finding/device/propagate_to_next_surface.hpp"
#include "traccc/finding/device/remove_duplicates.hpp"
#include "traccc/finding/finding_config.hpp"
#include "traccc/utils/logging.hpp"
#include "traccc/utils/memory_resource.hpp"
#include "traccc/utils/projections.hpp"
#include "traccc/utils/propagation.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

namespace traccc::alpaka::details {
namespace kernels {

/// Alpaka kernel functor for @c traccc::device::make_barcode_sequence
struct make_barcode_sequence {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        const device::make_barcode_sequence_payload payload) const {

        const device::global_index_t globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0];
        device::make_barcode_sequence(globalThreadIdx, payload);
    }
};

/// Alpaka kernel functor for @c traccc::device::apply_interaction
template <typename detector_t>
struct apply_interaction {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc, const finding_config cfg,
        const device::apply_interaction_payload<detector_t> payload) const {

        const device::global_index_t globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0];
        device::apply_interaction<detector_t>(globalThreadIdx, cfg, payload);
    }
};

/// Alpaka kernel functor for @c traccc::device::find_tracks
template <typename detector_t>
struct find_tracks {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc, const finding_config cfg,
        const device::find_tracks_payload<detector_t>* payload) const {

        auto& shared_num_out_params =
            ::alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);
        auto& shared_out_offset =
            ::alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);
        auto& shared_candidates_size =
            ::alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);
        unsigned long long int* const s =
            ::alpaka::getDynSharedMem<unsigned long long int>(acc);
        unsigned long long int* shared_insertion_mutex = s;

        alpaka::barrier<TAcc> barrier(&acc);
        details::thread_id1 thread_id(acc);

        const unsigned int blockDimX = thread_id.getBlockDimX();
        std::pair<unsigned int, unsigned int>* shared_candidates =
            reinterpret_cast<std::pair<unsigned int, unsigned int>*>(
                &shared_insertion_mutex[blockDimX]);

        device::find_tracks<detector_t>(
            thread_id, barrier, cfg, *payload,
            {shared_num_out_params, shared_out_offset, shared_insertion_mutex,
             shared_candidates, shared_candidates_size});
    }
};

/// Alpaka kernel functor for
/// @c traccc::device::fill_finding_duplicate_removal_sort_keys
struct fill_finding_duplicate_removal_sort_keys {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        const device::fill_finding_duplicate_removal_sort_keys_payload& payload)
        const {

        const device::global_index_t globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0];
        device::fill_finding_duplicate_removal_sort_keys(globalThreadIdx,
                                                         payload);
    }
};

/// Alpaka kernel functor for @c traccc::device::remove_duplicates
struct remove_duplicates {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc, const finding_config& cfg,
        const device::remove_duplicates_payload& payload) const {

        const device::global_index_t globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0];
        device::remove_duplicates(globalThreadIdx, cfg, payload);
    }
};

/// Alpaka kernel functor for
/// @c traccc::device::fill_finding_propagation_sort_keys
struct fill_finding_propagation_sort_keys {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        const device::fill_finding_propagation_sort_keys_payload payload)
        const {

        const device::global_index_t globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0];
        device::fill_finding_propagation_sort_keys(globalThreadIdx, payload);
    }
};

/// Alpaka kernel functor for @c traccc::device::propagate_to_next_surface
template <typename propagator_t, typename bfield_t>
struct propagate_to_next_surface {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc, const finding_config cfg,
        const device::propagate_to_next_surface_payload<propagator_t, bfield_t>*
            payload) const {

        device::global_index_t globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0];
        device::propagate_to_next_surface<propagator_t, bfield_t>(
            globalThreadIdx, cfg, *payload);
    }
};

/// Alpaka kernel functor for @c traccc::device::build_tracks
struct build_tracks {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc, const device::build_tracks_payload payload) const {

        device::global_index_t globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0];
        device::build_tracks(globalThreadIdx, payload);
    }
};

}  // namespace kernels

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
/// @param queue             The Alpaka queue to use
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
    const Logger& log, Queue& queue) {

    assert(config.min_step_length_for_next_surface >
               math::fabs(config.propagation.navigation.overstep_tolerance) &&
           "Min step length for the next surface should be higher than the "
           "overstep tolerance");

    // Create a logger.
    auto logger = [&log]() -> const Logger& { return log; };

    // Number of threads per block to use.
    const Idx threadsPerBlock = getWarpSize<Acc>() * 2;

    /*****************************************************************
     * Measurement Operations
     *****************************************************************/

    const measurement_collection_types::const_view::size_type n_measurements =
        copy.get_size(measurements);

    // Get copy of barcode uniques
    measurement_collection_types::buffer uniques_buffer{n_measurements,
                                                        mr.main};
    copy.setup(uniques_buffer)->wait();
    measurement_collection_types::device uniques(uniques_buffer);

    measurement_collection_types::device::iterator uniques_end =
        details::unique_copy(queue, mr, measurements.ptr(),
                             measurements.ptr() + n_measurements,
                             uniques.begin(), measurement_equal_comp());
    const unsigned int n_modules =
        static_cast<unsigned int>(uniques_end - uniques.begin());

    // Get upper bounds of unique elements
    vecmem::data::vector_buffer<unsigned int> upper_bounds_buffer{n_modules,
                                                                  mr.main};
    copy.setup(upper_bounds_buffer)->wait();
    vecmem::device_vector<unsigned int> upper_bounds(upper_bounds_buffer);

    details::upper_bound(queue, mr, measurements.ptr(),
                         measurements.ptr() + n_measurements, uniques.begin(),
                         uniques.begin() + n_modules, upper_bounds.begin(),
                         measurement_sort_comp());

    /*****************************************************************
     * Kernel1: Create barcode sequence
     *****************************************************************/

    vecmem::data::vector_buffer<detray::geometry::barcode> barcodes_buffer{
        n_modules, mr.main};
    copy.setup(barcodes_buffer)->wait();

    {
        Idx blocksPerGrid =
            (barcodes_buffer.size() + threadsPerBlock - 1) / threadsPerBlock;
        auto workDiv = makeWorkDiv<Acc>(blocksPerGrid, threadsPerBlock);

        ::alpaka::exec<Acc>(queue, workDiv, kernels::make_barcode_sequence{},
                            device::make_barcode_sequence_payload{
                                uniques_buffer, barcodes_buffer});
        ::alpaka::wait(queue);
    }

    const unsigned int n_seeds = copy.get_size(seeds);

    // Prepare input parameters with seeds
    bound_track_parameters_collection_types::buffer in_params_buffer(n_seeds,
                                                                     mr.main);
    copy.setup(in_params_buffer)->wait();
    copy(seeds, in_params_buffer, vecmem::copy::type::device_to_device)->wait();
    vecmem::data::vector_buffer<unsigned int> param_liveness_buffer(n_seeds,
                                                                    mr.main);
    copy.setup(param_liveness_buffer)->wait();
    copy.memset(param_liveness_buffer, 1)->wait();

    // Number of tracks per seed
    vecmem::data::vector_buffer<unsigned int> n_tracks_per_seed_buffer(n_seeds,
                                                                       mr.main);
    copy.setup(n_tracks_per_seed_buffer)->wait();

    // Create a buffer for links
    unsigned int link_buffer_capacity = config.initial_links_per_seed * n_seeds;
    vecmem::data::vector_buffer<candidate_link> links_buffer(
        link_buffer_capacity, mr.main, vecmem::data::buffer_type::resizable);
    copy.setup(links_buffer)->wait();

    // Create a buffer of tip links
    vecmem::data::vector_buffer<unsigned int> tips_buffer{
        config.max_num_branches_per_seed * n_seeds, mr.main,
        vecmem::data::buffer_type::resizable};
    copy.setup(tips_buffer)->wait();
    vecmem::data::vector_buffer<unsigned int> tip_length_buffer{
        config.max_num_branches_per_seed * n_seeds, mr.main};
    copy.setup(tip_length_buffer)->wait();

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
            Idx blocksPerGrid =
                (n_in_params + threadsPerBlock - 1) / threadsPerBlock;
            auto workDiv = makeWorkDiv<Acc>(blocksPerGrid, threadsPerBlock);

            ::alpaka::exec<Acc>(
                queue, workDiv, kernels::apply_interaction<detector_t>{},
                config,
                device::apply_interaction_payload<detector_t>{
                    det, n_in_params, in_params_buffer, param_liveness_buffer});
            ::alpaka::wait(queue);
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
        copy.setup(updated_params_buffer)->wait();

        vecmem::data::vector_buffer<unsigned int> updated_liveness_buffer(
            n_max_candidates, mr.main);
        copy.setup(updated_liveness_buffer)->wait();

        // Reset the number of tracks per seed
        copy.memset(n_tracks_per_seed_buffer, 0)->wait();

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

            copy.setup(new_links_buffer)->wait();
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
            // Now copy it to device memory.
            vecmem::data::vector_buffer<payload_t> device_payload(1u, mr.main);
            copy.setup(device_payload)->wait();
            copy(vecmem::data::vector_view<const payload_t>(1u, &host_payload),
                 device_payload)
                ->wait();

            // The number of threads to use per block in the track finding.
            const Idx blocksPerGrid =
                (n_in_params + threadsPerBlock - 1) / threadsPerBlock;
            const auto workDiv =
                makeWorkDiv<Acc>(blocksPerGrid, threadsPerBlock);

            // Submit the kernel to the queue.
            ::alpaka::exec<Acc>(queue, workDiv,
                                kernels::find_tracks<detector_t>{}, config,
                                device_payload.ptr());
            ::alpaka::wait(queue);

            std::swap(in_params_buffer, updated_params_buffer);
            std::swap(param_liveness_buffer, updated_liveness_buffer);

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
                const auto workDiv = makeWorkDiv<Acc>(nBlocks, nThreads);

                ::alpaka::exec<Acc>(
                    queue, workDiv,
                    kernels::fill_finding_duplicate_removal_sort_keys{},
                    device::fill_finding_duplicate_removal_sort_keys_payload{
                        .links_view = links_buffer,
                        .param_liveness_view = param_liveness_buffer,
                        .link_last_measurement_view =
                            link_last_measurement_buffer,
                        .param_ids_view = param_ids_buffer,
                        .n_links = n_candidates,
                        .curr_links_idx = step_to_link_idx_map[step],
                        .n_measurements = n_measurements});
                ::alpaka::wait(queue);
            }

            vecmem::device_vector<unsigned int> keys_device(
                link_last_measurement_buffer);
            vecmem::device_vector<unsigned int> param_ids_device(
                param_ids_buffer);
            details::sort_by_key(queue, mr, keys_device.begin(),
                                 keys_device.end(), param_ids_device.begin());

            /*
             * Then, we run the actual duplicate removal kernel.
             */
            {
                const unsigned int nThreads = 256;
                const unsigned int nBlocks =
                    (n_candidates + nThreads - 1) / nThreads;
                const auto workDiv = makeWorkDiv<Acc>(nBlocks, nThreads);

                ::alpaka::exec<Acc>(
                    queue, workDiv, kernels::remove_duplicates{}, config,
                    device::remove_duplicates_payload{
                        .links_view = links_buffer,
                        .link_last_measurement_view =
                            link_last_measurement_buffer,
                        .param_ids_view = param_ids_buffer,
                        .param_liveness_view = param_liveness_buffer,
                        .n_links = n_candidates,
                        .curr_links_idx = step_to_link_idx_map[step],
                        .n_measurements = n_measurements,
                        .step = step});
                ::alpaka::wait(queue);
            }
        }

        if (step == config.max_track_candidates_per_track - 1) {
            break;
        }

        if (n_candidates > 0) {
            /*****************************************************************
             * Kernel4: Get key and value for parameter sorting
             *****************************************************************/

            vecmem::data::vector_buffer<unsigned int> param_ids_buffer(
                n_candidates, mr.main);
            copy.setup(param_ids_buffer)->wait();

            {
                vecmem::data::vector_buffer<device::sort_key> keys_buffer(
                    n_candidates, mr.main);
                copy.setup(keys_buffer)->wait();

                Idx blocksPerGrid =
                    (n_candidates + threadsPerBlock - 1) / threadsPerBlock;
                auto workDiv = makeWorkDiv<Acc>(blocksPerGrid, threadsPerBlock);

                ::alpaka::exec<Acc>(
                    queue, workDiv,
                    kernels::fill_finding_propagation_sort_keys{},
                    device::fill_finding_propagation_sort_keys_payload{
                        in_params_buffer, param_liveness_buffer, keys_buffer,
                        param_ids_buffer});
                ::alpaka::wait(queue);

                // Sort the key and values
                vecmem::device_vector<device::sort_key> keys_device(
                    keys_buffer);
                vecmem::device_vector<unsigned int> param_ids_device(
                    param_ids_buffer);
                details::sort_by_key(queue, mr, keys_device.begin(),
                                     keys_device.end(),
                                     param_ids_device.begin());
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
                // Now copy it to device memory.
                vecmem::data::vector_buffer<payload_t> device_payload(1u,
                                                                      mr.main);
                copy.setup(device_payload)->wait();
                copy(vecmem::data::vector_view<const payload_t>(1u,
                                                                &host_payload),
                     device_payload)
                    ->wait();

                // The number of threads to use per block in the propagation.
                const Idx blocksPerGrid =
                    (n_candidates + threadsPerBlock - 1) / threadsPerBlock;
                const auto workDiv =
                    makeWorkDiv<Acc>(blocksPerGrid, threadsPerBlock);

                // Launch the kernel to propagate all active tracks to the next
                // surface.
                ::alpaka::exec<Acc>(
                    queue, workDiv,
                    kernels::propagate_to_next_surface<
                        traccc::details::ckf_propagator_t<detector_t, bfield_t>,
                        bfield_t>{},
                    config, device_payload.ptr());
                ::alpaka::wait(queue);
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
    auto n_tips_total = copy.get_size(tips_buffer);

    std::vector<unsigned int> tips_length_host;

    if (n_tips_total > 0) {
        copy(tip_length_buffer, tips_length_host)->wait();
        tips_length_host.resize(n_tips_total);
    }

    // Create track candidate buffer
    edm::track_candidate_collection<default_algebra>::buffer
        track_candidates_buffer{tips_length_host, mr.main, mr.host};
    copy.setup(track_candidates_buffer)->wait();

    if (n_tips_total > 0) {
        const Idx blocksPerGrid =
            (n_tips_total + threadsPerBlock - 1) / threadsPerBlock;
        const auto workDiv = makeWorkDiv<Acc>(blocksPerGrid, threadsPerBlock);

        ::alpaka::exec<Acc>(queue, workDiv, kernels::build_tracks{},
                            device::build_tracks_payload{
                                seeds,
                                links_buffer,
                                tips_buffer,
                                {track_candidates_buffer, measurements}});
        ::alpaka::wait(queue);
    }

    return track_candidates_buffer;
}

}  // namespace traccc::alpaka::details

namespace alpaka::trait {

/// Specify how much dynamic shared memory is needed for the
/// @c traccc::alpaka::details::kernels::find_tracks kernel.
template <typename TAcc, typename detector_t>
struct BlockSharedMemDynSizeBytes<
    traccc::alpaka::details::kernels::find_tracks<detector_t>, TAcc> {
    template <typename TVec, typename... TArgs>
    ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
        traccc::alpaka::details::kernels::find_tracks<
            detector_t> const& /* kernel */,
        TVec const& blockThreadExtent, TVec const& /* threadElemExtent */,
        TArgs const&... /* args */
        ) -> std::size_t {
        return static_cast<std::size_t>(blockThreadExtent.prod()) *
                   sizeof(unsigned long long int) +
               2 * static_cast<std::size_t>(blockThreadExtent.prod()) *
                   sizeof(std::pair<unsigned int, unsigned int>);
    }
};

}  // namespace alpaka::trait

namespace alpaka {

/// Convince Alpaka that
/// @c traccc::device::fill_finding_duplicate_removal_sort_keys_payload
/// is trivially copyable
template <>
struct IsKernelArgumentTriviallyCopyable<
    traccc::device::fill_finding_duplicate_removal_sort_keys_payload, void>
    : std::true_type {};

/// Convince Alpaka that
/// @c traccc::device::remove_duplicates_payload
/// is trivially copyable
template <>
struct IsKernelArgumentTriviallyCopyable<
    traccc::device::remove_duplicates_payload, void> : std::true_type {};

}  // namespace alpaka
