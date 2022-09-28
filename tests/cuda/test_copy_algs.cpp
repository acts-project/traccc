/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/device/container_d2h_copy_alg.hpp"
#include "traccc/device/container_h2d_copy_alg.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/utils/memory_resource.hpp"

// VecMem include(s).
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/cuda/copy.hpp>

// GoogleTest include(s).
#include <gtest/gtest.h>

GTEST_TEST(CUDAContainerCopy, CellHostToDeviceToHost) {

    // Create the memory resource(s).
    vecmem::host_memory_resource host_mr;
    vecmem::cuda::device_memory_resource device_mr;
    traccc::memory_resource mr{device_mr, &host_mr};

    // Create a cell container on the host.
    traccc::pixel_data pdata;
    traccc::cell_container_types::host host_orig{&host_mr};
    static constexpr std::size_t CONTAINER_SIZE = 100;
    host_orig.reserve(CONTAINER_SIZE);
    for (std::size_t i = 0; i < CONTAINER_SIZE; ++i) {
        host_orig.push_back(
            traccc::cell_container_types::host::header_type{
                0, i + 1, {}, 0, {5, 0}, {4, 0}, pdata},
            traccc::cell_container_types::host::vector_type<
                traccc::cell_container_types::host::item_type>{
                i + 1,
                {static_cast<traccc::channel_id>(i + 1),
                 static_cast<traccc::channel_id>(i + 2)},
                &host_mr});
    }

    // Construct the copy algorithm(s).
    vecmem::cuda::copy copy;
    traccc::device::container_h2d_copy_alg<traccc::cell_container_types> h2d{
        mr, copy};
    traccc::device::container_d2h_copy_alg<traccc::cell_container_types> d2h{
        mr, copy};

    // Copy the container to the device, and back.
    traccc::cell_container_types::host host_copy =
        d2h(h2d(traccc::get_data(host_orig)));

    // Compare the two host containers.
    EXPECT_EQ(host_orig.size(), host_copy.size());
    for (std::size_t i = 0; i < host_orig.size(); ++i) {
        const traccc::cell_container_types::host::element_view orig_view =
            host_orig[i];
        const traccc::cell_container_types::host::element_view copy_view =
            host_copy[i];
        EXPECT_EQ(orig_view.header, copy_view.header);
        EXPECT_EQ(orig_view.items.size(), copy_view.items.size());
        for (std::size_t j = 0; j < orig_view.items.size(); ++j) {
            EXPECT_EQ(orig_view.items[j], copy_view.items[j]);
        }
    }
}
