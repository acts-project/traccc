/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/data/jagged_vector_buffer.hpp"
#include "vecmem/containers/data/vector_buffer.hpp"
#include "vecmem/containers/device_vector.hpp"
#include "vecmem/containers/jagged_device_vector.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

namespace {

/// Helper function for comparing the contents of 1D device vectors
template <typename T>
void check_equal(const vecmem::device_vector<T>& v1,
                 const vecmem::device_vector<T>& v2) {

    ASSERT_EQ(v1.size(), v2.size());
    for (typename vecmem::device_vector<T>::size_type i = 0; i < v1.size();
         ++i) {
        EXPECT_EQ(v1[i], v2[i]);
    }
}

/// Helper function for comparing the contents of 1D device vectors
template <typename T>
void check_contains(const vecmem::device_vector<T>& small,
                    const vecmem::device_vector<T>& large) {

    ASSERT_LE(small.size(), large.size());
    for (typename vecmem::device_vector<T>::size_type i = 0; i < small.size();
         ++i) {
        EXPECT_EQ(small[i], large[i]);
    }
}

/// Helper function for comparing the contents of jagged device vectors
template <typename T>
void check_equal(const vecmem::jagged_device_vector<T>& v1,
                 const vecmem::jagged_device_vector<T>& v2) {

    ASSERT_EQ(v1.size(), v2.size());
    for (typename vecmem::device_vector<T>::size_type i = 0; i < v1.size();
         ++i) {
        check_equal(v1[i], v2[i]);
    }
}

/// Helper function for comparing the contents of jagged device vectors
template <typename T>
void check_contains(const vecmem::jagged_device_vector<T>& v1,
                    const vecmem::jagged_device_vector<T>& v2) {

    ASSERT_EQ(v1.size(), v2.size());
    for (typename vecmem::device_vector<T>::size_type i = 0; i < v1.size();
         ++i) {
        check_contains(v1[i], v2[i]);
    }
}

}  // namespace

void copy_tests::SetUp() {

    // Set up the reference data.
    m_ref = {{1, 5, 6, 74, 234, 43, 22}, host_mr_ptr()};
    m_jagged_ref = {{{{1, 2, 3, 4, 5}, host_mr_ptr()},
                     {{6, 7}, host_mr_ptr()},
                     {{8, 9, 10, 11}, host_mr_ptr()},
                     vecmem::vector<int>(host_mr_ptr()),
                     {{12, 13, 14, 15, 16, 17, 18}, host_mr_ptr()},
                     {{19}, host_mr_ptr()},
                     {{20}, host_mr_ptr()}},
                    host_mr_ptr()};
}

vecmem::copy& copy_tests::main_copy() {
    return *(std::get<0>(GetParam()));
}

vecmem::copy& copy_tests::host_copy() {
    return *(std::get<1>(GetParam()));
}

vecmem::memory_resource& copy_tests::main_mr() {
    return *(std::get<2>(GetParam()));
}

vecmem::memory_resource& copy_tests::host_mr() {
    return *(std::get<3>(GetParam()));
}

vecmem::memory_resource* copy_tests::host_mr_ptr() {
    return std::get<3>(GetParam());
}

vecmem::vector<int>& copy_tests::ref() {
    return m_ref;
}

const vecmem::vector<int>& copy_tests::cref() const {
    return m_ref;
}

vecmem::jagged_vector<int>& copy_tests::jagged_ref() {
    return m_jagged_ref;
}

const vecmem::jagged_vector<int>& copy_tests::jagged_cref() const {
    return m_jagged_ref;
}

/// Test for copying 1-dimensional vectors
TEST_P(copy_tests, vector) {

    // Create a view of its data.
    auto reference_data = vecmem::get_data(ref());

    // Make a copy of this reference.
    auto device_copy_data = main_copy().to(reference_data, main_mr());
    auto host_copy_data = main_copy().to(device_copy_data, host_mr());

    // Create device vectors over the two, and check them.
    vecmem::device_vector<int> reference_vector(reference_data);
    vecmem::device_vector<int> copy_vector(host_copy_data);
    check_equal(reference_vector, copy_vector);
}

/// Test for copying 1-dimensional (const) vectors
TEST_P(copy_tests, const_vector) {

    // Create a view of its data.
    const auto reference_data = vecmem::get_data(cref());

    // Make a copy of this reference.
    const auto device_copy_data = main_copy().to(reference_data, main_mr());
    const auto host_copy_data = main_copy().to(device_copy_data, host_mr());

    // Create device vectors over the two, and check them.
    vecmem::device_vector<const int> reference_vector(reference_data);
    vecmem::device_vector<const int> copy_vector(host_copy_data);
    check_equal(reference_vector, copy_vector);
}

/// Test for copying 1-dimensional, fixed size vector buffers
TEST_P(copy_tests, fixed_vector_buffer) {

    // Get the size of the reference vector.
    using vecmem_size_type = vecmem::device_vector<int>::size_type;
    const vecmem_size_type size = static_cast<vecmem_size_type>(cref().size());

    // Create non-resizable device and host buffers, with the "exact sizes".
    vecmem::data::vector_buffer<int> device_buffer(
        size, main_mr(), vecmem::data::buffer_type::fixed_size);
    main_copy().setup(device_buffer)->wait();
    vecmem::data::vector_buffer<int> host_buffer1(
        size, host_mr(), vecmem::data::buffer_type::fixed_size),
        host_buffer2(size, host_mr(), vecmem::data::buffer_type::fixed_size);
    host_copy().setup(host_buffer1)->wait();
    host_copy().setup(host_buffer2)->wait();

    // Copy data around.
    host_copy()(vecmem::get_data(cref()), host_buffer1,
                vecmem::copy::type::host_to_host)
        ->wait();
    main_copy()(host_buffer1, device_buffer, vecmem::copy::type::host_to_device)
        ->wait();
    main_copy()(device_buffer, host_buffer2, vecmem::copy::type::device_to_host)
        ->wait();

    // Check the results.
    vecmem::device_vector<const int> reference_vector(vecmem::get_data(cref()));
    vecmem::device_vector<const int> copy_vector(host_buffer2);
    check_equal(reference_vector, copy_vector);
}

/// Test for copying 1-dimensional, resizable vector buffers
TEST_P(copy_tests, resizable_vector_buffer) {

    // Get the size of the reference vector.
    using vecmem_size_type = vecmem::device_vector<int>::size_type;
    const vecmem_size_type size = static_cast<vecmem_size_type>(cref().size());

    // Create resizable device and host buffers, with the "exact sizes".
    vecmem::data::vector_buffer<int> device_buffer(
        size, main_mr(), vecmem::data::buffer_type::resizable);
    main_copy().setup(device_buffer)->wait();
    vecmem::data::vector_buffer<int> host_buffer1(
        size, host_mr(), vecmem::data::buffer_type::resizable),
        host_buffer2(size, host_mr(), vecmem::data::buffer_type::resizable);
    host_copy().setup(host_buffer1)->wait();
    host_copy().setup(host_buffer2)->wait();

    // Copy data around.
    host_copy()(vecmem::get_data(cref()), host_buffer1,
                vecmem::copy::type::host_to_host)
        ->wait();
    main_copy()(host_buffer1, device_buffer, vecmem::copy::type::host_to_device)
        ->wait();
    main_copy()(device_buffer, host_buffer2, vecmem::copy::type::device_to_host)
        ->wait();

    // Check the results.
    vecmem::device_vector<const int> reference_vector(vecmem::get_data(cref()));
    vecmem::device_vector<const int> copy_vector(host_buffer2);
    check_equal(reference_vector, copy_vector);
}

/// Test(s) for copying 1-dimensional, mismatched sized vector buffers
TEST_P(copy_tests, mismatched_vector_buffer) {

    // Get the size of the reference vector.
    using vecmem_size_type = vecmem::device_vector<int>::size_type;
    const vecmem_size_type size = static_cast<vecmem_size_type>(cref().size());

    // Create non-resizable device and host buffers, which are (progressively)
    // larger than needed.
    vecmem::data::vector_buffer<int> device_buffer1(
        size + 10u, main_mr(), vecmem::data::buffer_type::fixed_size);
    main_copy().setup(device_buffer1)->wait();
    vecmem::data::vector_buffer<int> host_buffer1(
        size, host_mr(), vecmem::data::buffer_type::fixed_size),
        host_buffer2(size + 20u, host_mr(),
                     vecmem::data::buffer_type::fixed_size);
    host_copy().setup(host_buffer1)->wait();
    host_copy().setup(host_buffer2)->wait();

    // Copy data around.
    host_copy()(vecmem::get_data(cref()), host_buffer1,
                vecmem::copy::type::host_to_host)
        ->wait();
    main_copy()(host_buffer1, device_buffer1,
                vecmem::copy::type::host_to_device)
        ->wait();
    main_copy()(device_buffer1, host_buffer2,
                vecmem::copy::type::device_to_host)
        ->wait();

    // Check the results.
    vecmem::device_vector<const int> reference_vector(vecmem::get_data(cref()));
    vecmem::device_vector<const int> copy_vector1(host_buffer2);
    check_contains(reference_vector, copy_vector1);

    // Create non-resizable device and host buffers, which are (progressively)
    // smaller than needed.
    vecmem::data::vector_buffer<int> device_buffer2(
        size - 1u, main_mr(), vecmem::data::buffer_type::fixed_size);
    main_copy().setup(device_buffer2)->wait();
    vecmem::data::vector_buffer<int> host_buffer3(
        size - 2u, host_mr(), vecmem::data::buffer_type::fixed_size);
    host_copy().setup(host_buffer3)->wait();

    // Verify that data cannot be copied around like this.
    EXPECT_THROW(main_copy()(host_buffer1, device_buffer2,
                             vecmem::copy::type::host_to_device)
                     ->wait(),
                 std::exception);
    EXPECT_THROW(main_copy()(device_buffer2, host_buffer3,
                             vecmem::copy::type::device_to_host)
                     ->wait(),
                 std::exception);

    // Create resizable device and host buffers, which are (progressively)
    // larger than needed.
    vecmem::data::vector_buffer<int> device_buffer3(
        size + 10u, main_mr(), vecmem::data::buffer_type::resizable);
    main_copy().setup(device_buffer3)->wait();
    vecmem::data::vector_buffer<int> host_buffer4(
        size + 20u, host_mr(), vecmem::data::buffer_type::resizable);
    host_copy().setup(host_buffer4)->wait();

    // Copy data around.
    main_copy()(host_buffer1, device_buffer3,
                vecmem::copy::type::host_to_device)
        ->wait();
    main_copy()(device_buffer3, host_buffer4,
                vecmem::copy::type::device_to_host)
        ->wait();

    // Check the results.
    vecmem::device_vector<const int> copy_vector2(host_buffer4);
    check_equal(reference_vector, copy_vector2);

    // Create resizable device buffer that's smaller than needed. Note that we
    // can't easily test a failure with D->H copy here. Since the H->D copy will
    // fail, the device buffer will remain at size 0. So the copy back to the
    // host will succeed. (With no copy actually happening.)
    vecmem::data::vector_buffer<int> device_buffer4(
        size - 1u, main_mr(), vecmem::data::buffer_type::resizable);
    main_copy().setup(device_buffer4)->wait();

    // Verify that data cannot be copied around like this.
    EXPECT_THROW(main_copy()(host_buffer1, device_buffer4,
                             vecmem::copy::type::host_to_device)
                     ->wait(),
                 std::exception);
}

/// Test for copying jagged vectors
TEST_P(copy_tests, jagged_vector) {

    // Create a view of its data.
    auto reference_data = vecmem::get_data(jagged_ref());

    // Make a copy of this reference.
    auto device_copy_data =
        main_copy().to(reference_data, main_mr(), host_mr_ptr());
    auto host_copy_data = main_copy().to(device_copy_data, host_mr());

    // Create device vectors over the two, and check them.
    vecmem::jagged_device_vector<int> reference_vector(reference_data);
    vecmem::jagged_device_vector<int> copy_vector(host_copy_data);
    check_equal(reference_vector, copy_vector);
}

/// Test for copying (const) jagged vectors
TEST_P(copy_tests, const_jagged_vector) {

    // Create a view of its data.
    const auto reference_data = vecmem::get_data(jagged_cref());

    // Make a copy of this reference.
    auto device_copy_data =
        main_copy().to(reference_data, main_mr(), host_mr_ptr());
    auto host_copy_data = main_copy().to(device_copy_data, host_mr());

    // Create device vectors over the two, and check them.
    vecmem::jagged_device_vector<const int> reference_vector(reference_data);
    vecmem::jagged_device_vector<const int> copy_vector(host_copy_data);
    check_equal(reference_vector, copy_vector);
}

/// Test for copying jagged, fixed size vector buffers
TEST_P(copy_tests, fixed_jagged_vector_buffer) {

    // Create a view of the reference data.
    const auto reference_data = vecmem::get_data(jagged_cref());

    // Create non-resizable device and host buffers, with the "exact sizes".
    vecmem::data::jagged_vector_buffer<int> device_buffer(
        reference_data, main_mr(), host_mr_ptr(),
        vecmem::data::buffer_type::fixed_size);
    main_copy().setup(device_buffer)->wait();
    vecmem::data::jagged_vector_buffer<int> host_buffer1(
        reference_data, host_mr(), nullptr,
        vecmem::data::buffer_type::fixed_size),
        host_buffer2(reference_data, host_mr(), nullptr,
                     vecmem::data::buffer_type::fixed_size);
    host_copy().setup(host_buffer1)->wait();
    host_copy().setup(host_buffer2)->wait();

    // Copy data around.
    host_copy()(reference_data, host_buffer1, vecmem::copy::type::host_to_host)
        ->wait();
    main_copy()(host_buffer1, device_buffer, vecmem::copy::type::host_to_device)
        ->wait();
    main_copy()(device_buffer, host_buffer2, vecmem::copy::type::device_to_host)
        ->wait();

    // Check the results.
    vecmem::jagged_device_vector<const int> reference_vector(reference_data);
    vecmem::jagged_device_vector<const int> copy_vector(host_buffer2);
    check_equal(reference_vector, copy_vector);
}

/// Test for copying jagged, resizable vector buffers
TEST_P(copy_tests, resizable_jagged_vector_buffer) {

    // Create a view of the reference data.
    const auto reference_data = vecmem::get_data(jagged_cref());

    // Create non-resizable device and host buffers, with the "exact sizes".
    vecmem::data::jagged_vector_buffer<int> device_buffer(
        reference_data, main_mr(), host_mr_ptr(),
        vecmem::data::buffer_type::resizable);
    main_copy().setup(device_buffer)->wait();
    vecmem::data::jagged_vector_buffer<int> host_buffer1(
        reference_data, host_mr(), nullptr,
        vecmem::data::buffer_type::resizable),
        host_buffer2(reference_data, host_mr(), nullptr,
                     vecmem::data::buffer_type::resizable);
    host_copy().setup(host_buffer1)->wait();
    host_copy().setup(host_buffer2)->wait();

    // Copy data around.
    host_copy()(reference_data, host_buffer1, vecmem::copy::type::host_to_host)
        ->wait();
    main_copy()(host_buffer1, device_buffer, vecmem::copy::type::host_to_device)
        ->wait();
    main_copy()(device_buffer, host_buffer2, vecmem::copy::type::device_to_host)
        ->wait();

    // Check the results.
    vecmem::jagged_device_vector<const int> reference_vector(reference_data);
    vecmem::jagged_device_vector<const int> copy_vector(host_buffer2);
    check_equal(reference_vector, copy_vector);
}

/// Test(s) for copying jagged, mismatched sized vector buffers
TEST_P(copy_tests, mismatched_jagged_vector_buffer) {

    // Create a view of the reference data.
    const auto reference_data = vecmem::get_data(jagged_cref());

    // Create non-resizable device and host buffers, which are (progressively)
    // larger than needed.
    std::vector<std::size_t> large_sizes1(jagged_cref().size()),
        large_sizes2(jagged_cref().size());
    std::transform(jagged_cref().begin(), jagged_cref().end(),
                   large_sizes1.begin(),
                   [](const auto& vec) { return vec.size() + 10u; });
    std::transform(jagged_cref().begin(), jagged_cref().end(),
                   large_sizes2.begin(),
                   [](const auto& vec) { return vec.size() + 20u; });
    vecmem::data::jagged_vector_buffer<int> device_buffer1(
        large_sizes1, main_mr(), host_mr_ptr(),
        vecmem::data::buffer_type::fixed_size);
    main_copy().setup(device_buffer1)->wait();
    vecmem::data::jagged_vector_buffer<int> host_buffer1(
        reference_data, host_mr(), nullptr,
        vecmem::data::buffer_type::fixed_size),
        host_buffer2(large_sizes2, host_mr(), nullptr,
                     vecmem::data::buffer_type::fixed_size);
    host_copy().setup(host_buffer1)->wait();
    host_copy().setup(host_buffer2)->wait();

    // Copy data around.
    host_copy()(reference_data, host_buffer1, vecmem::copy::type::host_to_host)
        ->wait();
    main_copy()(host_buffer1, device_buffer1,
                vecmem::copy::type::host_to_device)
        ->wait();
    main_copy()(device_buffer1, host_buffer2,
                vecmem::copy::type::device_to_host)
        ->wait();

    // Check the results.
    vecmem::jagged_device_vector<const int> reference_vector(reference_data);
    vecmem::jagged_device_vector<const int> copy_vector1(host_buffer2);
    check_contains(reference_vector, copy_vector1);

    // Create non-resizable device and host buffers, which are (progressively)
    // smaller than needed.
    std::vector<std::size_t> small_sizes1(jagged_cref().size()),
        small_sizes2(jagged_cref().size());
    std::transform(jagged_cref().begin(), jagged_cref().end(),
                   small_sizes1.begin(), [](const auto& vec) {
                       return (vec.size() >= 1u) ? vec.size() - 1u : 0u;
                   });
    std::transform(jagged_cref().begin(), jagged_cref().end(),
                   small_sizes2.begin(), [](const auto& vec) {
                       return (vec.size() >= 2u) ? vec.size() - 2u : 0u;
                   });
    vecmem::data::jagged_vector_buffer<int> device_buffer2(
        small_sizes1, main_mr(), host_mr_ptr(),
        vecmem::data::buffer_type::fixed_size);
    main_copy().setup(device_buffer1)->wait();
    vecmem::data::jagged_vector_buffer<int> host_buffer3(
        small_sizes2, host_mr(), nullptr,
        vecmem::data::buffer_type::fixed_size);
    host_copy().setup(host_buffer3)->wait();

    // Verify that data cannot be copied around like this.
    EXPECT_THROW(main_copy()(host_buffer1, device_buffer2,
                             vecmem::copy::type::host_to_device)
                     ->wait(),
                 std::exception);
    EXPECT_THROW(main_copy()(device_buffer2, host_buffer3,
                             vecmem::copy::type::device_to_host)
                     ->wait(),
                 std::exception);

    // Create resizable device and host buffers, which are (progressively)
    // larger than needed.
    vecmem::data::jagged_vector_buffer<int> device_buffer3(
        large_sizes1, main_mr(), host_mr_ptr(),
        vecmem::data::buffer_type::resizable);
    main_copy().setup(device_buffer3)->wait();
    vecmem::data::jagged_vector_buffer<int> host_buffer4(
        large_sizes2, host_mr(), nullptr, vecmem::data::buffer_type::resizable);
    host_copy().setup(host_buffer4)->wait();

    // Copy data around.
    main_copy()(host_buffer1, device_buffer3,
                vecmem::copy::type::host_to_device)
        ->wait();
    main_copy()(device_buffer3, host_buffer4,
                vecmem::copy::type::device_to_host)
        ->wait();

    // Check the results.
    vecmem::jagged_device_vector<const int> copy_vector2(host_buffer4);
    check_equal(reference_vector, copy_vector2);

    // Create resizable device buffer that's smaller than needed. Note that we
    // can't easily test a failure with D->H copy here. Since the H->D copy will
    // fail, the device buffer will remain at size(s) 0. So the copy back to the
    // host will succeed. (With no copy actually happening.)
    vecmem::data::jagged_vector_buffer<int> device_buffer4(
        small_sizes1, main_mr(), host_mr_ptr(),
        vecmem::data::buffer_type::resizable);
    main_copy().setup(device_buffer4)->wait();

    // Verify that data cannot be copied around like this.
    EXPECT_THROW(main_copy()(host_buffer1, device_buffer4,
                             vecmem::copy::type::host_to_device)
                     ->wait(),
                 std::exception);
}

/// Tests for @c vecmem::copy::memset
TEST_P(copy_tests, memset) {

    // Size for the 1-dimensional buffer(s).
    static const unsigned int BUFFER_SIZE = 10;

    // Test(s) with a 1-dimensional buffer.
    vecmem::data::vector_buffer<int> device_buffer1(BUFFER_SIZE, main_mr());
    main_copy().setup(device_buffer1)->wait();
    main_copy().memset(device_buffer1, 5)->wait();
    vecmem::vector<int> vector1(host_mr_ptr());
    main_copy()(device_buffer1, vector1)->wait();
    EXPECT_EQ(vector1.size(), BUFFER_SIZE);
    static const int REFERENCE = 0x05050505;
    for (int value : vector1) {
        EXPECT_EQ(value, REFERENCE);
    }

    vecmem::data::vector_buffer<std::tuple<unsigned int, float, double> >
        device_buffer2(BUFFER_SIZE, main_mr());
    main_copy().setup(device_buffer2)->wait();
    main_copy().memset(device_buffer2, 0)->wait();
    vecmem::vector<std::tuple<unsigned int, float, double> > vector2(
        host_mr_ptr());
    main_copy()(device_buffer2, vector2)->wait();
    EXPECT_EQ(vector2.size(), BUFFER_SIZE);
    for (const std::tuple<unsigned int, float, double>& value : vector2) {
        EXPECT_EQ(std::get<0>(value), 0u);
        EXPECT_EQ(std::get<1>(value), 0.f);
        EXPECT_EQ(std::get<2>(value), 0.);
    }

    // Size(s) for the jagged buffer(s).
    static const std::vector<std::size_t> JAGGED_BUFFER_SIZES = {3, 6, 6, 3, 0,
                                                                 2, 7, 2, 4, 0};

    // Test(s) with a jagged buffer.
    vecmem::data::jagged_vector_buffer<int> device_buffer3(
        JAGGED_BUFFER_SIZES, main_mr(), host_mr_ptr());
    main_copy().setup(device_buffer3)->wait();
    main_copy().memset(device_buffer3, 5)->wait();
    vecmem::jagged_vector<int> vector3(host_mr_ptr());
    main_copy()(device_buffer3, vector3)->wait();
    EXPECT_EQ(vector3.size(), JAGGED_BUFFER_SIZES.size());
    for (std::size_t i = 0; i < vector3.size(); ++i) {
        EXPECT_EQ(vector3.at(i).size(), JAGGED_BUFFER_SIZES.at(i));
        for (int value : vector3.at(i)) {
            EXPECT_EQ(value, REFERENCE);
        }
    }

    vecmem::data::jagged_vector_buffer<std::tuple<unsigned int, float, double> >
        device_buffer4(JAGGED_BUFFER_SIZES, main_mr(), host_mr_ptr());
    main_copy().setup(device_buffer4)->wait();
    main_copy().memset(device_buffer4, 0)->wait();
    vecmem::jagged_vector<std::tuple<unsigned int, float, double> > vector4(
        host_mr_ptr());
    main_copy()(device_buffer4, vector4)->wait();
    EXPECT_EQ(vector4.size(), JAGGED_BUFFER_SIZES.size());
    for (std::size_t i = 0; i < vector4.size(); ++i) {
        EXPECT_EQ(vector4.at(i).size(), JAGGED_BUFFER_SIZES.at(i));
        for (const std::tuple<unsigned int, float, double>& value :
             vector4.at(i)) {
            EXPECT_EQ(std::get<0>(value), 0u);
            EXPECT_EQ(std::get<1>(value), 0.f);
            EXPECT_EQ(std::get<2>(value), 0.);
        }
    }
}
