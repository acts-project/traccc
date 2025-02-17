/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "../sanity/contiguous_on.hpp"
#include "../utils/barrier.hpp"
#include "../utils/calculate1DimNdRange.hpp"
#include "../utils/global_index.hpp"
#include "../utils/thread_id.hpp"

// Project include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_candidate.hpp"
#include "traccc/finding/actors/ckf_aborter.hpp"
#include "traccc/finding/actors/interaction_register.hpp"
#include "traccc/finding/candidate_link.hpp"
#include "traccc/finding/device/apply_interaction.hpp"
#include "traccc/finding/device/build_tracks.hpp"
#include "traccc/finding/device/fill_sort_keys.hpp"
#include "traccc/finding/device/find_tracks.hpp"
#include "traccc/finding/device/make_barcode_sequence.hpp"
#include "traccc/finding/device/propagate_to_next_surface.hpp"
#include "traccc/finding/device/prune_tracks.hpp"
#include "traccc/finding/finding_config.hpp"
#include "traccc/utils/memory_resource.hpp"
#include "traccc/utils/projections.hpp"

// Detray include(s).
#include <detray/propagator/actors.hpp>
#include <detray/propagator/propagator.hpp>

// VecMem include(s).
#include <vecmem/utils/copy.hpp>
#include <vecmem/utils/sycl/local_accessor.hpp>

// oneDPL include(s).
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wshadow"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wshorten-64-to-32"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wimplicit-int-conversion"
#pragma clang diagnostic ignored "-Wimplicit-int-float-conversion"
#pragma clang diagnostic ignored "-Wsign-compare"
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#pragma clang diagnostic pop

// SYCL include(s).
#include <sycl/sycl.hpp>

namespace traccc::sycl::details {
namespace kernels {
struct make_barcode_sequence {};
template <typename T>
struct apply_interaction {};
template <typename T>
struct find_tracks {};
struct fill_sort_keys {};
template <typename T>
struct propagate_to_next_surface {};
struct build_tracks {};
struct prune_tracks {};
}  // namespace kernels

/// Templated implementation of the track finding algorithm.
///
/// Concrete track finding algorithms can use this function with the appropriate
/// specializations, to find tracks on top of a specific detector type, magnetic
/// field type, and track finding configuration.
///
/// @tparam stepper_t The stepper type used for the track propagation
/// @tparam navigator_t The navigator type used for the track navigation
/// @tparam kernel_t Structure to generate unique kernel names with
///
/// @param det               A view of the detector object
/// @param field             The magnetic field object
/// @param measurements_view All measurements in an event
/// @param seeds_view        All seeds in an event to start the track finding
///                          with
/// @param config            The track finding configuration
/// @param mr                The memory resource(s) to use
/// @param copy              The copy object to use
/// @param queue             The SYCL queue to use
///
/// @return A buffer of the found track candidates
///
template <typename stepper_t, typename navigator_t, typename kernel_t>
track_candidate_container_types::buffer find_tracks(
    const typename navigator_t::detector_type::view_type& det,
    const typename stepper_t::magnetic_field_type& field,
    const measurement_collection_types::const_view& measurements,
    const bound_track_parameters_collection_types::const_view& seeds,
    const finding_config& config, const memory_resource& mr, vecmem::copy& copy,
    ::sycl::queue& queue) {

    assert(is_contiguous_on<measurement_collection_types::const_device>(
        measurement_module_projection(), mr.main, copy, queue, measurements));

    // oneDPL policy to use, forcing execution onto the same device that the
    // hand-written kernels would run on.
    auto policy = oneapi::dpl::execution::device_policy{queue};

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
        oneapi::dpl::unique_copy(policy, measurements.ptr(),
                                 measurements.ptr() + n_measurements,
                                 uniques.begin(), measurement_equal_comp());
    const unsigned int n_modules =
        static_cast<unsigned int>(uniques_end - uniques.begin());

    // Get upper bounds of unique elements
    vecmem::data::vector_buffer<unsigned int> upper_bounds_buffer{n_modules,
                                                                  mr.main};
    copy.setup(upper_bounds_buffer)->wait();
    vecmem::device_vector<unsigned int> upper_bounds(upper_bounds_buffer);

    oneapi::dpl::upper_bound(policy, measurements.ptr(),
                             measurements.ptr() + n_measurements,
                             uniques.begin(), uniques.begin() + n_modules,
                             upper_bounds.begin(), measurement_sort_comp());
    queue.wait_and_throw();

    /*****************************************************************
     * Kernel1: Create barcode sequence
     *****************************************************************/

    vecmem::data::vector_buffer<detray::geometry::barcode> barcodes_buffer{
        n_modules, mr.main};
    copy.setup(barcodes_buffer)->wait();

    queue
        .submit([&](::sycl::handler& h) {
            h.parallel_for<kernels::make_barcode_sequence>(
                calculate1DimNdRange(n_modules, 64),
                [uniques_view = vecmem::get_data(uniques_buffer),
                 barcodes_view = vecmem::get_data(barcodes_buffer)](
                    ::sycl::nd_item<1> item) {
                    device::make_barcode_sequence(
                        details::global_index(item),
                        {uniques_view, barcodes_view});
                });
        })
        .wait_and_throw();

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

    // Create a map for links
    std::map<unsigned int, vecmem::data::vector_buffer<candidate_link>>
        link_map;

    // Create a buffer of tip links
    vecmem::data::vector_buffer<typename candidate_link::link_index_type>
        tips_buffer{config.max_num_branches_per_seed * n_seeds, mr.main,
                    vecmem::data::buffer_type::resizable};
    copy.setup(tips_buffer)->wait();

    // Link size
    std::vector<std::size_t> n_candidates_per_step;
    n_candidates_per_step.reserve(config.max_track_candidates_per_track);

    unsigned int n_in_params = n_seeds;
    for (unsigned int step = 0;
         step < config.max_track_candidates_per_track && n_in_params > 0;
         step++) {

        /*****************************************************************
         * Kernel2: Apply material interaction
         ****************************************************************/

        queue
            .submit([&](::sycl::handler& h) {
                h.parallel_for<kernels::apply_interaction<kernel_t>>(
                    calculate1DimNdRange(n_in_params, 64),
                    [config, det, n_in_params,
                     in_params = vecmem::get_data(in_params_buffer),
                     param_liveness = vecmem::get_data(param_liveness_buffer)](
                        ::sycl::nd_item<1> item) {
                        device::apply_interaction<
                            typename navigator_t::detector_type>(
                            details::global_index(item), config,
                            {det, n_in_params, in_params, param_liveness});
                    });
            })
            .wait_and_throw();

        /*****************************************************************
         * Kernel3: Find valid tracks
         *****************************************************************/

        // Previous step
        const unsigned int prev_step = (step == 0 ? 0 : step - 1);

        // Buffer for kalman-updated parameters spawned by the
        // measurement candidates
        const unsigned int n_max_candidates =
            n_in_params * config.max_num_branches_per_surface;

        bound_track_parameters_collection_types::buffer updated_params_buffer(
            n_in_params * config.max_num_branches_per_surface, mr.main);
        copy.setup(updated_params_buffer)->wait();

        vecmem::data::vector_buffer<unsigned int> updated_liveness_buffer(
            n_in_params * config.max_num_branches_per_surface, mr.main);
        copy.setup(updated_liveness_buffer)->wait();

        // Create the link map
        link_map[step] = {n_in_params * config.max_num_branches_per_surface,
                          mr.main};
        copy.setup(link_map[step])->wait();

        vecmem::unique_alloc_ptr<unsigned int> n_candidates_device =
            vecmem::make_unique_alloc<unsigned int>(mr.main);
        queue.memset(n_candidates_device.get(), 0, sizeof(unsigned int))
            .wait_and_throw();

        // The number of threads to use per block in the track finding.
        static const unsigned int nFindTracksThreads = 64;

        // Submit the kernel to the queue.
        queue
            .submit([&](::sycl::handler& h) {
                // Allocate shared memory for the kernel.
                vecmem::sycl::local_accessor<unsigned int>
                    shared_num_candidates(nFindTracksThreads, h);
                vecmem::sycl::local_accessor<
                    std::pair<unsigned int, unsigned int>>
                    shared_candidates(2 * nFindTracksThreads, h);
                vecmem::sycl::local_accessor<unsigned int>
                    shared_candidates_size(1, h);

                // Launch the kernel.
                h.parallel_for<kernels::find_tracks<kernel_t>>(
                    calculate1DimNdRange(n_in_params, nFindTracksThreads),
                    [config, det, measurements,
                     in_params = vecmem::get_data(in_params_buffer),
                     param_liveness = vecmem::get_data(param_liveness_buffer),
                     n_in_params, barcodes = vecmem::get_data(barcodes_buffer),
                     upper_bounds = vecmem::get_data(upper_bounds_buffer),
                     previous_candidate_links =
                         vecmem::get_data(link_map.at(prev_step)),
                     step, n_max_candidates,
                     updated_params = vecmem::get_data(updated_params_buffer),
                     updated_liveness =
                         vecmem::get_data(updated_liveness_buffer),
                     current_candidate_links =
                         vecmem::get_data(link_map.at(step)),
                     n_candidates = n_candidates_device.get(),
                     shared_candidates_size, shared_num_candidates,
                     shared_candidates](::sycl::nd_item<1> item) {
                        // SYCL wrappers used in the algorithm.
                        const details::barrier barrier{item};
                        const details::thread_id thread_id{item};

                        // Call the device function to find tracks.
                        device::find_tracks<
                            std::decay_t<typename navigator_t::detector_type>>(
                            thread_id, barrier, config,
                            {det, measurements, in_params, param_liveness,
                             n_in_params, barcodes, upper_bounds,
                             previous_candidate_links, step, n_max_candidates,
                             updated_params, updated_liveness,
                             current_candidate_links, n_candidates},
                            {&(shared_num_candidates[0]),
                             &(shared_candidates[0]),
                             shared_candidates_size[0]});
                    });
            })
            .wait_and_throw();

        std::swap(in_params_buffer, updated_params_buffer);
        std::swap(param_liveness_buffer, updated_liveness_buffer);

        // Get the number of candidates back to the host.
        unsigned int n_candidates = 0;
        queue
            .memcpy(&n_candidates, n_candidates_device.get(),
                    sizeof(unsigned int))
            .wait_and_throw();

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

                queue
                    .submit([&](::sycl::handler& h) {
                        h.parallel_for<kernels::fill_sort_keys>(
                            calculate1DimNdRange(n_candidates, 256),
                            [in_params = vecmem::get_data(in_params_buffer),
                             keys = vecmem::get_data(keys_buffer),
                             param_ids = vecmem::get_data(param_ids_buffer)](
                                ::sycl::nd_item<1> item) {
                                device::fill_sort_keys(
                                    details::global_index(item),
                                    {in_params, keys, param_ids});
                            });
                    })
                    .wait_and_throw();

                // Sort the keys and values.
                vecmem::device_vector<device::sort_key> keys_device(
                    keys_buffer);
                vecmem::device_vector<unsigned int> param_ids_device(
                    param_ids_buffer);
                oneapi::dpl::sort_by_key(policy, keys_device.begin(),
                                         keys_device.end(),
                                         param_ids_device.begin());
                queue.wait_and_throw();
            }

            /*****************************************************************
             * Kernel5: Propagate to the next surface
             *****************************************************************/

            // Reset the number of tracks per seed
            copy.memset(n_tracks_per_seed_buffer, 0)->wait();

            /// Actor types
            using algebra_type =
                typename navigator_t::detector_type::algebra_type;
            using scalar_type =
                typename navigator_t::detector_type::scalar_type;
            using interactor_type =
                detray::pointwise_material_interactor<algebra_type>;
            using actor_type =
                detray::actor_chain<detray::pathlimit_aborter<scalar_type>,
                                    detray::parameter_transporter<algebra_type>,
                                    interaction_register<interactor_type>,
                                    interactor_type, ckf_aborter>;
            using propagator_type =
                detray::propagator<stepper_t, navigator_t, actor_type>;

            // Launch the kernel to propagate all active tracks to the next
            // surface.
            queue
                .submit([&](::sycl::handler& h) {
                    h.parallel_for<
                        kernels::propagate_to_next_surface<kernel_t>>(
                        calculate1DimNdRange(n_candidates, 64),
                        [config, det, field,
                         in_params = vecmem::get_data(in_params_buffer),
                         param_liveness =
                             vecmem::get_data(param_liveness_buffer),
                         param_ids = vecmem::get_data(param_ids_buffer),
                         current_candidate_links =
                             vecmem::get_data(link_map.at(step)),
                         step, n_candidates,
                         tips = vecmem::get_data(tips_buffer),
                         n_tracks_per_seed =
                             vecmem::get_data(n_tracks_per_seed_buffer)](
                            ::sycl::nd_item<1> item) {
                            device::propagate_to_next_surface<
                                propagator_type,
                                typename stepper_t::magnetic_field_type>(
                                details::global_index(item), config,
                                {det, field, in_params, param_liveness,
                                 param_ids, current_candidate_links, step,
                                 n_candidates, tips, n_tracks_per_seed});
                        });
                })
                .wait_and_throw();
        }

        // Fill the candidate size vector
        n_candidates_per_step.push_back(n_candidates);

        n_in_params = n_candidates;
    }

    // Create link buffer
    vecmem::data::jagged_vector_buffer<candidate_link> links_buffer(
        n_candidates_per_step, mr.main, mr.host);
    copy.setup(links_buffer)->wait();

    // Copy link map to link buffer
    for (unsigned int it = 0;
         it < static_cast<unsigned int>(n_candidates_per_step.size()); it++) {

        vecmem::device_vector<candidate_link> in(link_map.at(it));
        vecmem::device_vector<candidate_link> out(
            *(links_buffer.host_ptr() + it));

        oneapi::dpl::copy(policy, in.begin(),
                          in.begin() + n_candidates_per_step[it], out.begin());
    }
    queue.wait_and_throw();

    /*****************************************************************
     * Kernel6: Build tracks
     *****************************************************************/

    // Get the number of tips
    auto n_tips_total = copy.get_size(tips_buffer);

    // Create track candidate buffer
    track_candidate_container_types::buffer track_candidates_buffer{
        {n_tips_total, mr.main},
        {std::vector<std::size_t>(n_tips_total,
                                  config.max_track_candidates_per_track),
         mr.main, mr.host, vecmem::data::buffer_type::resizable}};
    copy.setup(track_candidates_buffer.headers)->wait();
    copy.setup(track_candidates_buffer.items)->wait();
    track_candidate_container_types::view track_candidates =
        track_candidates_buffer;

    // Create buffer for valid indices
    vecmem::data::vector_buffer<unsigned int> valid_indices_buffer(n_tips_total,
                                                                   mr.main);
    copy.setup(valid_indices_buffer)->wait();

    unsigned int n_valid_tracks = 0u;

    if (n_tips_total > 0) {
        vecmem::unique_alloc_ptr<unsigned int> n_valid_tracks_device =
            vecmem::make_unique_alloc<unsigned int>(mr.main);
        queue.memset(n_valid_tracks_device.get(), 0, sizeof(unsigned int))
            .wait_and_throw();

        queue
            .submit([&](::sycl::handler& h) {
                h.parallel_for<kernels::build_tracks>(
                    calculate1DimNdRange(n_tips_total, 64),
                    [config, measurements, seeds,
                     links = vecmem::get_data(links_buffer),
                     tips = vecmem::get_data(tips_buffer), track_candidates,
                     valid_indices = vecmem::get_data(valid_indices_buffer),
                     n_valid_tracks =
                         n_valid_tracks_device.get()](::sycl::nd_item<1> item) {
                        device::build_tracks(
                            details::global_index(item), config,
                            {measurements, seeds, links, tips, track_candidates,
                             valid_indices, n_valid_tracks});
                    });
            })
            .wait_and_throw();

        queue
            .memcpy(&n_valid_tracks, n_valid_tracks_device.get(),
                    sizeof(unsigned int))
            .wait_and_throw();
    }

    // Create pruned candidate buffer
    track_candidate_container_types::buffer prune_candidates_buffer{
        {n_valid_tracks, mr.main},
        {std::vector<std::size_t>(n_valid_tracks,
                                  config.max_track_candidates_per_track),
         mr.main, mr.host, vecmem::data::buffer_type::resizable}};
    copy.setup(prune_candidates_buffer.headers)->wait();
    copy.setup(prune_candidates_buffer.items)->wait();
    track_candidate_container_types::view prune_candidates =
        prune_candidates_buffer;

    if (n_valid_tracks > 0) {

        queue
            .submit([&](::sycl::handler& h) {
                h.parallel_for<kernels::prune_tracks>(
                    calculate1DimNdRange(n_valid_tracks, 64),
                    [track_candidates,
                     valid_indices = vecmem::get_data(valid_indices_buffer),
                     prune_candidates](::sycl::nd_item<1> item) {
                        device::prune_tracks(details::global_index(item),
                                             {track_candidates, valid_indices,
                                              prune_candidates});
                    });
            })
            .wait_and_throw();
    }

    return prune_candidates_buffer;
}

}  // namespace traccc::sycl::details
