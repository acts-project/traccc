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
#include "../utils/oneDPL.hpp"
#include "../utils/thread_id.hpp"

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
#include "traccc/utils/memory_resource.hpp"
#include "traccc/utils/projections.hpp"
#include "traccc/utils/propagation.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>
#include <vecmem/utils/sycl/local_accessor.hpp>

// SYCL include(s).
#include <sycl/sycl.hpp>

namespace traccc::sycl::details {
namespace kernels {
template <typename T>
struct make_barcode_sequence {};
template <typename T>
struct apply_interaction {};
template <typename T>
struct find_tracks {};
template <typename T>
struct fill_finding_duplicate_removal_sort_keys {};
template <typename T>
struct remove_duplicates {};
template <typename T>
struct fill_finding_propagation_sort_keys {};
template <typename T>
struct propagate_to_next_surface {};
template <typename T>
struct build_tracks {};
}  // namespace kernels

/// Templated implementation of the track finding algorithm.
///
/// Concrete track finding algorithms can use this function with the appropriate
/// specializations, to find tracks on top of a specific detector type, magnetic
/// field type, and track finding configuration.
///
/// @tparam kernel_t   Structure to generate unique kernel names with
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
/// @param queue             The SYCL queue to use
///
/// @return A buffer of the found track candidates
///
template <typename kernel_t, typename detector_t, typename bfield_t>
edm::track_candidate_collection<default_algebra>::buffer
combinatorial_kalman_filter(
    const typename detector_t::const_view_type& det, const bfield_t& field,
    const measurement_collection_types::const_view& measurements,
    const bound_track_parameters_collection_types::const_view& seeds,
    const finding_config& config, const memory_resource& mr, vecmem::copy& copy,
    ::sycl::queue& queue) {

    assert(config.min_step_length_for_next_surface >
               math::fabs(config.propagation.navigation.overstep_tolerance) &&
           "Min step length for the next surface should be higher than the "
           "overstep tolerance");

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
            h.parallel_for<kernels::make_barcode_sequence<kernel_t>>(
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

        queue
            .submit([&](::sycl::handler& h) {
                h.parallel_for<kernels::apply_interaction<kernel_t>>(
                    calculate1DimNdRange(n_in_params, 64),
                    [config, det, n_in_params,
                     in_params = vecmem::get_data(in_params_buffer),
                     param_liveness = vecmem::get_data(param_liveness_buffer)](
                        ::sycl::nd_item<1> item) {
                        device::apply_interaction<detector_t>(
                            details::global_index(item), config,
                            {det, n_in_params, in_params, param_liveness});
                    });
            })
            .wait_and_throw();

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
            copy.setup(tmp_links_buffer)->wait();
            bound_track_parameters_collection_types::buffer tmp_params_buffer(
                n_max_candidates, mr.main);
            copy.setup(tmp_params_buffer)->wait();

            // The number of threads to use per block in the track finding.
            static const unsigned int nFindTracksThreads = 64;

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

            // Submit the kernel to the queue.
            queue
                .submit([&](::sycl::handler& h) {
                    // Allocate shared memory for the kernel.
                    vecmem::sycl::local_accessor<unsigned long long int>
                        shared_insertion_mutex(nFindTracksThreads, h);
                    vecmem::sycl::local_accessor<
                        std::pair<unsigned int, unsigned int>>
                        shared_candidates(2 * nFindTracksThreads, h);
                    vecmem::sycl::local_accessor<unsigned int>
                        shared_candidates_size(1, h);
                    vecmem::sycl::local_accessor<unsigned int>
                        shared_num_out_params(1, h);
                    vecmem::sycl::local_accessor<unsigned int>
                        shared_out_offset(1, h);
                    // Launch the kernel.
                    h.parallel_for<kernels::find_tracks<kernel_t>>(
                        calculate1DimNdRange(n_in_params, nFindTracksThreads),
                        [config, payload = device_payload.ptr(),
                         shared_insertion_mutex, shared_candidates,
                         shared_candidates_size, shared_num_out_params,
                         shared_out_offset](::sycl::nd_item<1> item) {
                            // SYCL wrappers used in the algorithm.
                            const details::barrier barrier{item};
                            const details::thread_id thread_id{item};

                            // Call the device function to find tracks.
                            device::find_tracks<detector_t>(
                                thread_id, barrier, config, *payload,
                                {shared_num_out_params[0], shared_out_offset[0],
                                 &(shared_insertion_mutex[0]),
                                 &(shared_candidates[0]),
                                 shared_candidates_size[0]});
                        });
                })
                .wait_and_throw();

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
            queue
                .submit([&](::sycl::handler& h) {
                    h.parallel_for<
                        kernels::fill_finding_duplicate_removal_sort_keys<
                            kernel_t>>(
                        calculate1DimNdRange(n_candidates, 256),
                        [links_view = vecmem::get_data(links_buffer),
                         param_liveness_view =
                             vecmem::get_data(param_liveness_buffer),
                         link_last_measurement_view =
                             vecmem::get_data(link_last_measurement_buffer),
                         param_ids_view = vecmem::get_data(param_ids_buffer),
                         n_candidates,
                         curr_links_idx = step_to_link_idx_map[step],
                         n_measurements](::sycl::nd_item<1> item) {
                            device::fill_finding_duplicate_removal_sort_keys(
                                details::global_index(item),
                                {.links_view = links_view,
                                 .param_liveness_view = param_liveness_view,
                                 .link_last_measurement_view =
                                     link_last_measurement_view,
                                 .param_ids_view = param_ids_view,
                                 .n_links = n_candidates,
                                 .curr_links_idx = curr_links_idx,
                                 .n_measurements = n_measurements});
                        });
                })
                .wait_and_throw();

            vecmem::device_vector<unsigned int> keys_device(
                link_last_measurement_buffer);
            vecmem::device_vector<unsigned int> param_ids_device(
                param_ids_buffer);
            oneapi::dpl::sort_by_key(policy, keys_device.begin(),
                                     keys_device.end(),
                                     param_ids_device.begin());
            queue.wait_and_throw();

            /*
             * Then, we run the actual duplicate removal kernel.
             */
            queue
                .submit([&](::sycl::handler& h) {
                    h.parallel_for<kernels::remove_duplicates<kernel_t>>(
                        calculate1DimNdRange(n_candidates, 256),
                        [config, links_view = vecmem::get_data(links_buffer),
                         link_last_measurement_view =
                             vecmem::get_data(link_last_measurement_buffer),
                         param_ids_view = vecmem::get_data(param_ids_buffer),
                         param_liveness_view =
                             vecmem::get_data(param_liveness_buffer),
                         n_candidates,
                         curr_links_idx = step_to_link_idx_map[step],
                         n_measurements, step](::sycl::nd_item<1> item) {
                            device::remove_duplicates(
                                details::global_index(item), config,
                                {.links_view = links_view,
                                 .link_last_measurement_view =
                                     link_last_measurement_view,
                                 .param_ids_view = param_ids_view,
                                 .param_liveness_view = param_liveness_view,
                                 .n_links = n_candidates,
                                 .curr_links_idx = curr_links_idx,
                                 .n_measurements = n_measurements,
                                 .step = step});
                        });
                })
                .wait_and_throw();
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

                queue
                    .submit([&](::sycl::handler& h) {
                        h.parallel_for<
                            kernels::fill_finding_propagation_sort_keys<
                                kernel_t>>(
                            calculate1DimNdRange(n_candidates, 256),
                            [in_params = vecmem::get_data(in_params_buffer),
                             param_liveness =
                                 vecmem::get_data(param_liveness_buffer),
                             keys = vecmem::get_data(keys_buffer),
                             param_ids = vecmem::get_data(param_ids_buffer)](
                                ::sycl::nd_item<1> item) {
                                device::fill_finding_propagation_sort_keys(
                                    details::global_index(item),
                                    {in_params, param_liveness, keys,
                                     param_ids});
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

                // Launch the kernel to propagate all active tracks to the next
                // surface.
                queue
                    .submit([&](::sycl::handler& h) {
                        h.parallel_for<
                            kernels::propagate_to_next_surface<kernel_t>>(
                            calculate1DimNdRange(n_candidates, 64),
                            [config, payload = device_payload.ptr()](
                                ::sycl::nd_item<1> item) {
                                device::propagate_to_next_surface<
                                    traccc::details::ckf_propagator_t<
                                        detector_t, bfield_t>,
                                    bfield_t>(details::global_index(item),
                                              config, *payload);
                            });
                    })
                    .wait_and_throw();
            }
        }

        n_in_params = n_candidates;
    }

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
        queue
            .submit([&](::sycl::handler& h) {
                h.parallel_for<kernels::build_tracks<kernel_t>>(
                    calculate1DimNdRange(n_tips_total, 64),
                    [measurements, seeds,
                     links = vecmem::get_data(links_buffer),
                     tips = vecmem::get_data(tips_buffer),
                     track_candidates = vecmem::get_data(
                         track_candidates_buffer)](::sycl::nd_item<1> item) {
                        device::build_tracks(
                            details::global_index(item),
                            {seeds,
                             links,
                             tips,
                             {track_candidates, measurements}});
                    });
            })
            .wait_and_throw();
    }

    return track_candidates_buffer;
}

}  // namespace traccc::sycl::details
