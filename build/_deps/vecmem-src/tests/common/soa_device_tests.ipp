/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "jagged_soa_container_helpers.hpp"
#include "simple_soa_container_helpers.hpp"

template <typename CONTAINER>
void soa_device_tests_base<CONTAINER>::modify_managed(
    const soa_device_test_parameters& params) {

    // Extract the needed parameters.
    vecmem::memory_resource& host_mr = std::get<0>(params);
    vecmem::memory_resource& managed_mr = std::get<2>(params);
    bool (*device_modify)(typename CONTAINER::view) =
        reinterpret_cast<bool (*)(typename CONTAINER::view)>(
            std::get<5>(params));

    // Create two host containers in host and managed memory.
    typename CONTAINER::host container1{host_mr};
    typename CONTAINER::host container2{managed_mr};

    // Fill them with some data.
    vecmem::testing::fill(container1);
    vecmem::testing::fill(container2);

    // Modify the first container, using a simple for loop.
    auto data1 = vecmem::get_data(container1);
    typename CONTAINER::device device1{data1};
    for (unsigned int i = 0; i < container1.size(); ++i) {
        vecmem::testing::modify(i, device1);
    }

    // Run a kernel that executes the modify function on the second container.
    if ((*device_modify)(vecmem::get_data(container2)) == false) {
        GTEST_SKIP();
    }

    // Compare the two.
    vecmem::testing::compare(vecmem::get_data(container1),
                             vecmem::get_data(container2));
}

template <typename CONTAINER>
void soa_device_tests_base<CONTAINER>::modify_device(
    const soa_device_test_parameters& params) {

    // Extract the needed parameters.
    vecmem::memory_resource& host_mr = std::get<0>(params);
    vecmem::memory_resource& device_mr = std::get<1>(params);
    vecmem::copy& copy = std::get<3>(params);
    bool (*device_modify)(typename CONTAINER::view) =
        reinterpret_cast<bool (*)(typename CONTAINER::view)>(
            std::get<5>(params));

    // Create a host container in host memory as a start.
    typename CONTAINER::host container1{host_mr};

    // Fill it with some data.
    vecmem::testing::fill(container1);

    // Copy it to the device.
    typename CONTAINER::buffer buffer;
    vecmem::testing::make_buffer(buffer, device_mr, host_mr,
                                 vecmem::data::buffer_type::fixed_size);
    copy.setup(buffer)->wait();
    copy(vecmem::get_data(container1), buffer,
         vecmem::copy::type::host_to_device)
        ->wait();

    // Modify the container in host memory, using a simple for loop.
    auto data1 = vecmem::get_data(container1);
    typename CONTAINER::device device1{data1};
    for (unsigned int i = 0; i < container1.size(); ++i) {
        vecmem::testing::modify(i, device1);
    }

    // Run a kernel that executes the modify function on the device buffer.
    if ((*device_modify)(buffer) == false) {
        GTEST_SKIP();
    }

    // Copy the data back to the host.
    typename CONTAINER::host container2{host_mr};
    copy(buffer, container2)->wait();

    // Compare the two.
    vecmem::testing::compare(vecmem::get_data(container1),
                             vecmem::get_data(container2));
}

template <typename CONTAINER>
void soa_device_tests_base<CONTAINER>::fill_device(
    const soa_device_test_parameters& params) {

    // Extract the needed parameters.
    vecmem::memory_resource& host_mr = std::get<0>(params);
    vecmem::memory_resource& device_mr = std::get<1>(params);
    vecmem::copy& copy = std::get<3>(params);
    bool (*device_fill)(typename CONTAINER::view) =
        reinterpret_cast<bool (*)(typename CONTAINER::view)>(
            std::get<4>(params));

    // Create a host buffer, and fill it.
    typename CONTAINER::buffer buffer1;
    vecmem::testing::make_buffer(buffer1, host_mr, host_mr,
                                 vecmem::data::buffer_type::resizable);
    copy.setup(buffer1)->wait();
    typename CONTAINER::device device1{buffer1};
    for (unsigned int i = 0; i < device1.capacity(); ++i) {
        vecmem::testing::fill(i, device1);
    }

    // Create a resizable device buffer, and fill it on the device.
    typename CONTAINER::buffer buffer2;
    vecmem::testing::make_buffer(buffer2, device_mr, host_mr,
                                 vecmem::data::buffer_type::resizable);
    copy.setup(buffer2)->wait();

    // Run a kernel that fills the buffer.
    if ((*device_fill)(buffer2) == false) {
        GTEST_SKIP();
    }

    // Copy the device buffer back to the host.
    typename CONTAINER::buffer buffer3;
    vecmem::testing::make_buffer(buffer3, host_mr, host_mr,
                                 vecmem::data::buffer_type::resizable);
    copy.setup(buffer3)->wait();
    copy(buffer2, buffer3, vecmem::copy::type::device_to_host)->wait();

    // Compare the two containers.
    vecmem::testing::compare(buffer1, buffer3);
}

TEST_P(soa_device_tests_simple, modify_managed) {

    modify_managed(GetParam());
}

TEST_P(soa_device_tests_simple, modify_device) {

    modify_device(GetParam());
}

TEST_P(soa_device_tests_simple, fill_device) {

    fill_device(GetParam());
}

TEST_P(soa_device_tests_jagged, modify_managed) {

    modify_managed(GetParam());
}

TEST_P(soa_device_tests_jagged, modify_device) {

    modify_device(GetParam());
}

TEST_P(soa_device_tests_jagged, fill_device) {

    fill_device(GetParam());
}
