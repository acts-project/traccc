/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "../utils/calculate1DimNdRange.hpp"
#include "../utils/global_index.hpp"
#include "../utils/oneDPL.hpp"

// Project include(s).
#include "traccc/edm/device/sort_key.hpp"
#include "traccc/edm/track_candidate_container.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/fitting/details/kalman_fitting_types.hpp"
#include "traccc/fitting/device/fill_sort_keys.hpp"
#include "traccc/fitting/device/fit.hpp"
#include "traccc/fitting/device/fit_backward.hpp"
#include "traccc/fitting/device/fit_forward.hpp"
#include "traccc/fitting/device/fit_prelude.hpp"
#include "traccc/fitting/fitting_config.hpp"
#include "traccc/utils/memory_resource.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

// SYCL include(s).
#include <sycl/sycl.hpp>

namespace traccc::sycl {
namespace kernels {

template <typename T>
struct fill_sort_keys;
template <typename T>
struct fit_prelude;
template <typename T>
struct fit_forward;
template <typename T>
struct fit_backward;

}  // namespace kernels

namespace details {

/// Templated implementation of the SYCL track fitting algorithm.
///
/// @tparam kernel_t   Structure to generate unique kernel names with
/// @tparam detector_t The (device) detector type to use
/// @tparam bfield_t   The magnetic field type to use
///
/// @param[in] det_view     A view of the detector geometry
/// @param[in] field_view   A view of the magnetic field
/// @param[in] track_candidates_view All track candidates to fit
/// @param[in] config       The fitting configuration
/// @param[in] mr           Memory resource(s) to use
/// @param[in] copy         The copy object to use for memory transfers
/// @param[in] queue        The SYCL queue to use for execution
///
/// @return A container of the fitted track states
///
template <typename kernel_t, typename detector_t, typename bfield_t>
track_state_container_types::buffer kalman_fitting(
    const typename detector_t::view_type& det_view, const bfield_t& field_view,
    const typename edm::track_candidate_container<
        typename detector_t::algebra_type>::const_view& track_candidates_view,
    const fitting_config& config, const memory_resource& mr, vecmem::copy& copy,
    ::sycl::queue& queue) {

    // Get the number of tracks.
    const edm::track_candidate_collection<
        default_algebra>::const_device::size_type n_tracks =
        copy.get_size(track_candidates_view.tracks);

    // Get the sizes of the track candidates in each track.
    const std::vector<unsigned int> candidate_sizes =
        copy.get_sizes(track_candidates_view.tracks);

    // Create the result buffer.
    track_state_container_types::buffer track_states_buffer{
        {n_tracks, mr.main},
        {candidate_sizes, mr.main, mr.host,
         vecmem::data::buffer_type::resizable}};
    vecmem::copy::event_type track_states_headers_setup_event =
        copy.setup(track_states_buffer.headers);
    vecmem::copy::event_type track_states_items_setup_event =
        copy.setup(track_states_buffer.items);

    // Return early, if there are no tracks.
    if (n_tracks == 0) {
        track_states_headers_setup_event->wait();
        track_states_items_setup_event->wait();
        return track_states_buffer;
    }

    std::vector<unsigned int> seqs_sizes(candidate_sizes.size());
    std::transform(candidate_sizes.begin(), candidate_sizes.end(),
                   seqs_sizes.begin(), [&config](const unsigned int sz) {
                       return std::max(sz * config.barcode_sequence_size_factor,
                                       config.min_barcode_sequence_capacity);
                   });
    vecmem::data::jagged_vector_buffer<detray::geometry::barcode> seqs_buffer{
        seqs_sizes, mr.main, mr.host, vecmem::data::buffer_type::resizable};
    copy.setup(seqs_buffer)->wait();

    // Create the buffers for sorting the parameter IDs.
    vecmem::data::vector_buffer<device::sort_key> keys_buffer(n_tracks,
                                                              mr.main);
    vecmem::data::vector_buffer<unsigned int> param_ids_buffer(n_tracks,
                                                               mr.main);
    vecmem::data::vector_buffer<unsigned int> param_liveness_buffer(n_tracks,
                                                                    mr.main);
    vecmem::copy::event_type keys_setup_event = copy.setup(keys_buffer);
    vecmem::copy::event_type param_ids_setup_event =
        copy.setup(param_ids_buffer);
    vecmem::copy::event_type param_liveness_setup_event =
        copy.setup(param_liveness_buffer);
    keys_setup_event->wait();
    param_ids_setup_event->wait();
    param_liveness_setup_event->wait();

    // The execution range for the two kernels of the function.
    static constexpr unsigned int localSize = 64;
    ::sycl::nd_range<1> range = calculate1DimNdRange(n_tracks, localSize);

    // Fill the keys and param_ids buffers.
    ::sycl::event fill_keys_event = queue.submit([&](::sycl::handler& h) {
        h.parallel_for<kernels::fill_sort_keys<kernel_t>>(
            range,
            [track_candidates_view, keys_view = vecmem::get_data(keys_buffer),
             param_ids_view =
                 vecmem::get_data(param_ids_buffer)](::sycl::nd_item<1> item) {
                device::fill_sort_keys(details::global_index(item),
                                       track_candidates_view.tracks, keys_view,
                                       param_ids_view);
            });
    });

    // Sort the key to get the sorted parameter ids
    vecmem::device_vector<device::sort_key> keys_device(keys_buffer);
    vecmem::device_vector<unsigned int> param_ids_device(param_ids_buffer);
    fill_keys_event.wait_and_throw();
    oneapi::dpl::sort_by_key(oneapi::dpl::execution::dpcpp_default,
                             keys_device.begin(), keys_device.end(),
                             param_ids_device.begin());

    // Run the fitting, using the sorted parameter IDs.
    track_state_container_types::view track_states_view = track_states_buffer;
    track_states_headers_setup_event->wait();
    track_states_items_setup_event->wait();

    queue
        .submit([&](::sycl::handler& h) {
            h.parallel_for<kernels::fit_prelude<kernel_t>>(
                range, [param_ids_view = vecmem::get_data(param_ids_buffer),
                        track_candidates_view, track_states_view,
                        param_liveness_view = vecmem::get_data(
                            param_liveness_buffer)](::sycl::nd_item<1> item) {
                    device::fit_prelude(details::global_index(item),
                                        param_ids_view, track_candidates_view,
                                        track_states_view, param_liveness_view);
                });
        })
        .wait_and_throw();

    using fitter_t = traccc::details::kalman_fitter_t<detector_t, bfield_t>;
    device::fit_payload<fitter_t> payload{
        .det_data = det_view,
        .field_data = field_view,
        .param_ids_view = param_ids_buffer,
        .param_liveness_view = param_liveness_buffer,
        .track_states_view = track_states_view,
        .barcodes_view = seqs_buffer};

    for (std::size_t i = 0; i < config.n_iterations; ++i) {
        queue
            .submit([&](::sycl::handler& h) {
                h.parallel_for<kernels::fit_forward<kernel_t>>(
                    range, [config, payload](::sycl::nd_item<1> item) {
                        device::fit_forward<fitter_t>(
                            details::global_index(item), config, payload);
                    });
            })
            .wait_and_throw();

        queue
            .submit([&](::sycl::handler& h) {
                h.parallel_for<kernels::fit_backward<kernel_t>>(
                    range, [config, payload](::sycl::nd_item<1> item) {
                        device::fit_backward<fitter_t>(
                            details::global_index(item), config, payload);
                    });
            })
            .wait_and_throw();
    }

    // Return the fitted tracks.
    return track_states_buffer;
}

}  // namespace details
}  // namespace traccc::sycl
