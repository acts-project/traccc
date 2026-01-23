/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "../utils/calculate1DimNdRange.hpp"
#include "../utils/get_queue.hpp"
#include "../utils/global_index.hpp"

// Project include(s).
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/seeding/device/form_spacepoints.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

// SYCL include(s).
#include <sycl/sycl.hpp>

namespace traccc::sycl::details {

/// Common implementation for the spacepoint formation algorithm's execute
/// functions
///
/// @tparam detector_t The detector type to use
///
/// @param det_view          The view of the detector to use
/// @param measurements_view The view of the measurements to process
/// @param mr                The memory resource to create the output with
/// @param copy              The copy object to use for the output buffer
/// @param queue             The queue to use for the computation
/// @return A buffer of the created spacepoints
///
template <typename detector_t>
edm::spacepoint_collection::buffer silicon_pixel_spacepoint_formation(
    const typename detector_t::view& det_view,
    const typename edm::measurement_collection<
        typename detector_t::device::algebra_type>::const_view&
        measurements_view,
    vecmem::memory_resource& mr, vecmem::copy& copy, ::sycl::queue& queue) {

    // Get the number of measurements.
    const unsigned int n_measurements = copy.get_size(measurements_view);
    if (n_measurements == 0) {
        return {};
    }

    // Create the result buffer.
    edm::spacepoint_collection::buffer result(
        n_measurements, mr, vecmem::data::buffer_type::resizable);
    vecmem::copy::event_type spacepoints_setup_event = copy.setup(result);

    // Calculate the range to run the spacepoint formation for.
    static constexpr unsigned int localSize = 32 * 2;
    const ::sycl::nd_range countRange =
        calculate1DimNdRange(n_measurements, localSize);

    // Put the detector view into device memory.
    vecmem::data::vector_buffer<typename detector_t::view> device_det_view(1u,
                                                                           mr);
    copy.setup(device_det_view)->wait();
    vecmem::copy::event_type detector_setup_event =
        copy(vecmem::data::vector_view<const typename detector_t::view>(
                 1u, &det_view),
             device_det_view);

    // Wait for the buffers to be ready.
    spacepoints_setup_event->wait();
    detector_setup_event->wait();

    // Run the spacepoint formation on the device.
    queue
        .submit([&](::sycl::handler& h) {
            h.parallel_for(countRange, [det_view_ptr = device_det_view.ptr(),
                                        measurements_view,
                                        spacepoints_view = vecmem::get_data(
                                            result)](::sycl::nd_item<1> item) {
                device::form_spacepoints<detector_t>(
                    details::global_index(item), *det_view_ptr,
                    measurements_view, spacepoints_view);
            });
        })
        .wait_and_throw();

    // Return the spacepoint buffer.
    return result;
}

}  // namespace traccc::sycl::details
