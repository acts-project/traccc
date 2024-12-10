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
vecmem::copy& soa_copy_tests_base<CONTAINER>::main_copy() {
    return *(std::get<0>(GetParam()));
}

template <typename CONTAINER>
vecmem::copy& soa_copy_tests_base<CONTAINER>::host_copy() {
    return *(std::get<1>(GetParam()));
}

template <typename CONTAINER>
vecmem::memory_resource& soa_copy_tests_base<CONTAINER>::main_mr() {
    return *(std::get<2>(GetParam()));
}

template <typename CONTAINER>
vecmem::memory_resource& soa_copy_tests_base<CONTAINER>::host_mr() {
    return *(std::get<3>(GetParam()));
}

template <typename CONTAINER>
void soa_copy_tests_base<CONTAINER>::host_to_fixed_device_to_host_direct() {

    // Create the "input" host container.
    typename CONTAINER::host input{host_mr()};

    // Fill it with some data.
    vecmem::testing::fill(input);

    // Create the (fixed sized) device buffer.
    typename CONTAINER::buffer device_buffer;
    vecmem::testing::make_buffer(device_buffer, main_mr(), host_mr(),
                                 vecmem::data::buffer_type::fixed_size);
    main_copy().setup(device_buffer)->wait();

    // Copy the data to the device.
    const typename CONTAINER::data input_data = vecmem::get_data(input);
    main_copy()(vecmem::get_data(input_data), device_buffer,
                vecmem::copy::type::host_to_device)
        ->wait();

    // Check the size of the device buffer.
    EXPECT_EQ(input.size(), main_copy().get_size(device_buffer));

    // Create the target host container.
    typename CONTAINER::host target{host_mr()};

    // Copy the data back to the host.
    main_copy()(device_buffer, target, vecmem::copy::type::device_to_host)
        ->wait();

    // Compare the two.
    vecmem::testing::compare(vecmem::get_data(input), vecmem::get_data(target));
}

template <typename CONTAINER>
void soa_copy_tests_base<CONTAINER>::host_to_fixed_device_to_host_optimal() {

    // Create the "input" host container.
    typename CONTAINER::host input{host_mr()};

    // Fill it with some data.
    vecmem::testing::fill(input);

    // Create a (fixed sized) host buffer, to stage the data into.
    typename CONTAINER::buffer host_buffer1;
    vecmem::testing::make_buffer(host_buffer1, host_mr(), host_mr(),
                                 vecmem::data::buffer_type::fixed_size);
    host_copy().setup(host_buffer1)->wait();

    // Stage the data into the host buffer.
    host_copy()(vecmem::get_data(input), host_buffer1)->wait();

    // Create the (fixed sized) device buffer.
    typename CONTAINER::buffer device_buffer;
    vecmem::testing::make_buffer(device_buffer, main_mr(), host_mr(),
                                 vecmem::data::buffer_type::fixed_size);
    main_copy().setup(device_buffer)->wait();

    // Copy the data from the host buffer to the device buffer.
    main_copy()(host_buffer1, device_buffer, vecmem::copy::type::host_to_device)
        ->wait();

    // Create a (fixed sized) host buffer, to stage the data back into.
    typename CONTAINER::buffer host_buffer2;
    vecmem::testing::make_buffer(host_buffer2, host_mr(), host_mr(),
                                 vecmem::data::buffer_type::fixed_size);
    host_copy().setup(host_buffer2)->wait();

    // Copy the data from the device buffer to the host buffer.
    main_copy()(device_buffer, host_buffer2, vecmem::copy::type::device_to_host)
        ->wait();

    // Create the target host container.
    typename CONTAINER::host target{host_mr()};

    // Copy the data from the host buffer to the target.
    host_copy()(host_buffer2, target)->wait();

    // Compare the relevant objects.
    vecmem::testing::compare(vecmem::get_data(input),
                             vecmem::get_data(host_buffer2));
    vecmem::testing::compare(vecmem::get_data(input), vecmem::get_data(target));
}

template <typename CONTAINER>
void soa_copy_tests_base<CONTAINER>::host_to_resizable_device_to_host() {

    // Create the "input" host container.
    typename CONTAINER::host input{host_mr()};

    // Fill it with some data.
    vecmem::testing::fill(input);

    // Create the (resizable) device buffer.
    typename CONTAINER::buffer device_buffer;
    vecmem::testing::make_buffer(device_buffer, main_mr(), host_mr(),
                                 vecmem::data::buffer_type::resizable);
    main_copy().setup(device_buffer)->wait();

    // Copy the data to the device.
    const typename CONTAINER::data input_data = vecmem::get_data(input);
    main_copy()(vecmem::get_data(input_data), device_buffer,
                vecmem::copy::type::host_to_device)
        ->wait();

    // Check the size of the device buffer.
    EXPECT_EQ(input.size(), main_copy().get_size(device_buffer));

    // Create the target host container.
    typename CONTAINER::host target{host_mr()};

    // Copy the data back to the host.
    main_copy()(device_buffer, target, vecmem::copy::type::device_to_host)
        ->wait();

    // Compare the two.
    vecmem::testing::compare(vecmem::get_data(input), vecmem::get_data(target));
}

template <typename CONTAINER>
void soa_copy_tests_base<
    CONTAINER>::host_to_fixed_device_to_resizable_device_to_host() {

    // Create the "input" host container.
    typename CONTAINER::host input{host_mr()};

    // Fill it with some data.
    vecmem::testing::fill(input);

    // Create the (fixed sized) device buffer.
    typename CONTAINER::buffer device_buffer1;
    vecmem::testing::make_buffer(device_buffer1, main_mr(), host_mr(),
                                 vecmem::data::buffer_type::fixed_size);
    main_copy().setup(device_buffer1)->wait();

    // Copy the data to the device.
    main_copy()(vecmem::get_data(input), device_buffer1,
                vecmem::copy::type::host_to_device)
        ->wait();

    // Check the size of the device buffer.
    EXPECT_EQ(input.size(), main_copy().get_size(device_buffer1));

    // Create the (resizable) device buffer.
    typename CONTAINER::buffer device_buffer2;
    vecmem::testing::make_buffer(device_buffer2, main_mr(), host_mr(),
                                 vecmem::data::buffer_type::resizable);

    // Copy the data from the fixed sized device buffer to the resizable one.
    main_copy()(device_buffer1, device_buffer2,
                vecmem::copy::type::device_to_device)
        ->wait();

    // Check the size of the device buffer.
    EXPECT_EQ(input.size(), main_copy().get_size(device_buffer2));

    // Create the target host container.
    typename CONTAINER::host target{host_mr()};

    // Copy the data back to the host.
    main_copy()(device_buffer2, target, vecmem::copy::type::device_to_host)
        ->wait();

    // Compare the two.
    vecmem::testing::compare(vecmem::get_data(input), vecmem::get_data(target));
}

TEST_P(soa_copy_tests_simple, host_to_fixed_device_to_host_direct) {

    host_to_fixed_device_to_host_direct();
}

TEST_P(soa_copy_tests_simple, host_to_fixed_device_to_host_optimal) {

    host_to_fixed_device_to_host_optimal();
}

TEST_P(soa_copy_tests_simple, host_to_resizable_device_to_host) {

    host_to_resizable_device_to_host();
}

TEST_P(soa_copy_tests_simple,
       host_to_fixed_device_to_resizable_device_to_host) {

    host_to_fixed_device_to_resizable_device_to_host();
}

TEST_P(soa_copy_tests_jagged, host_to_fixed_device_to_host_direct) {

    host_to_fixed_device_to_host_direct();
}

TEST_P(soa_copy_tests_jagged, host_to_fixed_device_to_host_optimal) {

    host_to_fixed_device_to_host_optimal();
}

TEST_P(soa_copy_tests_jagged, host_to_resizable_device_to_host) {

    host_to_resizable_device_to_host();
}

TEST_P(soa_copy_tests_jagged,
       host_to_fixed_device_to_resizable_device_to_host) {

    host_to_fixed_device_to_resizable_device_to_host();
}
