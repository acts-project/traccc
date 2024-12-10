/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/containers/data/jagged_vector_buffer.hpp"
#include "vecmem/containers/data/jagged_vector_view.hpp"
#include "vecmem/containers/data/vector_buffer.hpp"
#include "vecmem/containers/data/vector_view.hpp"
#include "vecmem/containers/device_vector.hpp"
#include "vecmem/containers/jagged_device_vector.hpp"
#include "vecmem/containers/jagged_vector.hpp"
#include "vecmem/containers/vector.hpp"
#include "vecmem/memory/contiguous_memory_resource.hpp"
#include "vecmem/memory/host_memory_resource.hpp"
#include "vecmem/utils/copy.hpp"
#include "vecmem/utils/type_traits.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <cstring>
#include <type_traits>

/// Test case for the custom device container types
class core_device_container_test : public testing::Test {

protected:
    /// Memory resource for the test(s)
    vecmem::host_memory_resource m_resource;
    /// Helper object for the memory copies.
    vecmem::copy m_copy;

};  // class core_device_container_test

/// Test that the "simple" data types are trivially constructible.
TEST_F(core_device_container_test, trivial_construct) {

    EXPECT_TRUE(std::is_trivially_default_constructible<
                vecmem::data::jagged_vector_view<const int> >());
    EXPECT_TRUE(std::is_trivially_constructible<
                vecmem::data::jagged_vector_view<const int> >());
    EXPECT_TRUE(std::is_trivially_copy_constructible<
                vecmem::data::jagged_vector_view<const int> >());

    EXPECT_TRUE(std::is_trivially_default_constructible<
                vecmem::data::jagged_vector_view<int> >());
    EXPECT_TRUE(std::is_trivially_constructible<
                vecmem::data::jagged_vector_view<int> >());
    EXPECT_TRUE(std::is_trivially_copy_constructible<
                vecmem::data::jagged_vector_view<int> >());

    EXPECT_TRUE(std::is_trivially_default_constructible<
                vecmem::data::vector_view<const int> >());
    EXPECT_TRUE(std::is_trivially_constructible<
                vecmem::data::vector_view<const int> >());
    EXPECT_TRUE(std::is_trivially_copy_constructible<
                vecmem::data::vector_view<const int> >());

    EXPECT_TRUE(std::is_trivially_default_constructible<
                vecmem::data::vector_view<int> >());
    EXPECT_TRUE(
        std::is_trivially_constructible<vecmem::data::vector_view<int> >());
    EXPECT_TRUE(std::is_trivially_copy_constructible<
                vecmem::data::vector_view<int> >());

    EXPECT_TRUE(
        std::is_default_constructible<vecmem::data::vector_buffer<int> >());
    EXPECT_TRUE(std::is_default_constructible<
                vecmem::data::jagged_vector_buffer<int> >());
    EXPECT_TRUE(std::is_default_constructible<
                vecmem::data::jagged_vector_data<int> >());
}

/// Test(s) for @c vecmem::data::vector_buffer
TEST_F(core_device_container_test, vector_buffer) {

    // Create a dummy vector in regular host memory.
    std::vector<int> host_vector{1, 2, 3, 4, 5};
    auto host_data = vecmem::get_data(host_vector);

    // Create an "owning copy" of the host vector's memory.
    vecmem::data::vector_buffer<int> device_data(host_data.size(), m_resource);
    m_copy(vecmem::get_data(host_vector), device_data)->wait();

    // Do some basic tests.
    EXPECT_EQ(device_data.size(), host_vector.size());
    EXPECT_EQ(memcmp(host_data.ptr(), device_data.ptr(),
                     host_data.size() * sizeof(int)),
              0);
}

/// Test(s) for @c vecmem::data::jagged_vector_buffer
TEST_F(core_device_container_test, jagged_vector_buffer) {

    // Create a dummy jagged vector in regular host memory.
    std::vector<std::vector<int> > host_vector{{},
                                               {1, 2, 3, 4, 5},
                                               {6, 7},
                                               {8, 9, 10, 11},
                                               {12, 13, 14, 15, 16, 17, 18},
                                               {},
                                               {},
                                               {19, 20},
                                               {},
                                               {21}};
    auto host_data = vecmem::get_data(host_vector, &m_resource);

    // Set up an "alternative" memory resource for the test.
    vecmem::contiguous_memory_resource cresource(m_resource, 16384);

    // Create a buffer to hold the same data.
    vecmem::data::jagged_vector_buffer<int> device_data1(host_data, m_resource);
    vecmem::data::jagged_vector_buffer<int> device_data2(host_data, m_resource,
                                                         &cresource);

    // Test the internal state of the buffer.
    EXPECT_EQ(device_data1.ptr(), device_data1.host_ptr());
    EXPECT_EQ(device_data1.size(), host_vector.size());
    EXPECT_NE(device_data2.ptr(), device_data2.host_ptr());
    EXPECT_EQ(device_data2.size(), host_vector.size());
    for (std::size_t i = 0; i < host_vector.size(); ++i) {
        EXPECT_EQ(device_data1.host_ptr()[i].size(), host_vector[i].size());
        EXPECT_EQ(device_data2.host_ptr()[i].size(), host_vector[i].size());
    }
    for (std::size_t i = 0; i < host_vector.size() - 1; ++i) {
        EXPECT_EQ(device_data1.host_ptr()[i].ptr() + host_vector[i].size(),
                  device_data1.host_ptr()[i + 1].ptr());
        EXPECT_EQ(device_data2.host_ptr()[i].ptr() + host_vector[i].size(),
                  device_data2.host_ptr()[i + 1].ptr());
    }
}

/// Test(s) for a "resizable" @c vecmem::data::vector_buffer
TEST_F(core_device_container_test, resizable_vector_buffer) {

    // Create an input vector in regular host memory.
    std::vector<int> host_vector{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // Helper local type definitions.
    typedef
        typename vecmem::data::vector_buffer<int>::size_type buffer_size_type;
    typedef typename vecmem::device_vector<int>::size_type vector_size_type;

    // Create a resizable buffer from that data.
    static constexpr buffer_size_type BUFFER_SIZE = 100;
    vecmem::data::vector_buffer<int> resizable_buffer(
        BUFFER_SIZE, m_resource, vecmem::data::buffer_type::resizable);
    m_copy.setup(resizable_buffer)->wait();
    EXPECT_EQ(resizable_buffer.capacity(), BUFFER_SIZE);
    m_copy(vecmem::get_data(host_vector), resizable_buffer)->wait();
    EXPECT_EQ(resizable_buffer.size(), host_vector.size());

    // Create a "device vector" on top of that resizable data.
    vecmem::device_vector<int> device_vector(resizable_buffer);

    // Perform some simple tests as a start.
    EXPECT_EQ(device_vector.size(), host_vector.size());
    EXPECT_EQ(device_vector.capacity(), BUFFER_SIZE);
    EXPECT_EQ(device_vector.max_size(), BUFFER_SIZE);
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(device_vector[static_cast<vector_size_type>(i)], i + 1);
    }

    // Modify the device vector in different ways, and check that it would work
    // as expected.
    device_vector.clear();
    EXPECT_EQ(device_vector.size(), static_cast<vector_size_type>(0));

    device_vector.push_back(10);
    EXPECT_EQ(device_vector.size(), static_cast<vector_size_type>(1));
    EXPECT_EQ(device_vector.at(0), 10);

    device_vector.emplace_back(15);
    EXPECT_EQ(device_vector.size(), static_cast<vector_size_type>(2));
    EXPECT_EQ(device_vector.back(), 15);

    device_vector.assign(20, 123);
    EXPECT_EQ(device_vector.size(), static_cast<vector_size_type>(20));

    for (int value : device_vector) {
        EXPECT_EQ(value, 123);
    }

    auto n1 = device_vector.bulk_append_implicit(2);
    EXPECT_EQ(device_vector.size(), static_cast<vector_size_type>(22));
    for (decltype(n1) i = 0; i < 2; ++i) {
        device_vector.at(n1 + i) = 123;
    }

    auto n2 = device_vector.bulk_append_implicit_unsafe(2);
    EXPECT_EQ(device_vector.size(), static_cast<vector_size_type>(24));
    for (decltype(n2) i = 0; i < 2; ++i) {
        device_vector.at(n2 + i) = 123;
    }

    auto n3 = device_vector.bulk_append(2);
    EXPECT_EQ(device_vector.size(), static_cast<vector_size_type>(26));
    for (decltype(n3) i = 0; i < 2; ++i) {
        device_vector.at(n3 + i) = 123;
    }

    device_vector.bulk_append(3, 123);
    EXPECT_EQ(device_vector.size(), static_cast<vector_size_type>(29));

    EXPECT_EQ(device_vector.size(), static_cast<vector_size_type>(29));
    for (int value : device_vector) {
        EXPECT_EQ(value, 123);
    }

    device_vector.resize(40, 234);
    EXPECT_EQ(device_vector.size(), static_cast<vector_size_type>(40));
    for (vector_size_type i = 0; i < 29; ++i) {
        EXPECT_EQ(device_vector[i], 123);
    }
    for (vector_size_type i = 29; i < 40; ++i) {
        EXPECT_EQ(device_vector[i], 234);
    }
    device_vector.resize(35);
    EXPECT_EQ(device_vector.size(), static_cast<vector_size_type>(35));
    for (vector_size_type i = 0; i < 29; ++i) {
        EXPECT_EQ(device_vector[i], 123);
    }
    for (vector_size_type i = 29; i < 35; ++i) {
        EXPECT_EQ(device_vector[i], 234);
    }

    device_vector.pop_back();
    EXPECT_EQ(device_vector.size(), static_cast<vector_size_type>(34));
    for (vector_size_type i = 0; i < 29; ++i) {
        EXPECT_EQ(device_vector[i], 123);
    }
    for (vector_size_type i = 29; i < 34; ++i) {
        EXPECT_EQ(device_vector[i], 234);
    }

    // Copy the modified data back into the "host vector", and check if that
    // succeeded.
    m_copy(resizable_buffer, host_vector)->wait();
    EXPECT_EQ(host_vector.size(), 34);
    for (std::size_t i = 0; i < 29; ++i) {
        EXPECT_EQ(host_vector[i], 123);
    }
    for (std::size_t i = 29; i < 34; ++i) {
        EXPECT_EQ(host_vector[i], 234);
    }
}

/// Test(s) for a "resizable" @c vecmem::data::vector_buffer with non-trivial
/// types
TEST_F(core_device_container_test, resizable_vector_buffer_nontrivial) {
#ifdef VECMEM_HAVE_IS_IMPLICIT_LIFETIME
    class nontrivial_class {
    public:
        nontrivial_class() = delete;
        nontrivial_class(const nontrivial_class&) = delete;
        nontrivial_class(nontrivial_class&& o) { m_value = o.m_value; };

        nontrivial_class(int v) : m_value(v) {}

        int get() { return m_value; }

        nontrivial_class& operator=(int v) {
            m_value = v;
            return *this;
        }

    private:
        int m_value;
    };

    static_assert(!vecmem::details::is_implicit_lifetime_v<nontrivial_class>,
                  "This test is designed to ensure that the type "
                  "`nontrivial_class` is not an implicit lifetime type.");

    // Helper local type definitions.
    typedef typename vecmem::data::vector_buffer<nontrivial_class>::size_type
        buffer_size_type;
    typedef typename vecmem::device_vector<nontrivial_class>::size_type
        vector_size_type;

    // Create a resizable buffer from that data.
    static constexpr buffer_size_type BUFFER_SIZE = 100;
    vecmem::data::vector_buffer<nontrivial_class> resizable_buffer(
        BUFFER_SIZE, m_resource, vecmem::data::buffer_type::resizable);
    m_copy.setup(resizable_buffer)->wait();
    EXPECT_EQ(resizable_buffer.capacity(), BUFFER_SIZE);
    EXPECT_EQ(resizable_buffer.size(), 0);

    // Create a "device vector" on top of that resizable data.
    vecmem::device_vector<nontrivial_class> device_vector(resizable_buffer);

    // Perform some simple tests as a start.
    EXPECT_EQ(device_vector.size(), 0);
    EXPECT_EQ(device_vector.capacity(), BUFFER_SIZE);
    EXPECT_EQ(device_vector.max_size(), BUFFER_SIZE);

    device_vector.resize_implicit_unsafe(BUFFER_SIZE);
    for (decltype(device_vector)::size_type i = 0; i < BUFFER_SIZE; ++i) {
        device_vector.at(i) = -100;
    }
    device_vector.resize_implicit_unsafe(10);
    EXPECT_EQ(device_vector.size(), 10);

    for (decltype(device_vector)::size_type i = 0; i < 10; ++i) {
        device_vector.at(i) = static_cast<int>(i) + static_cast<int>(1);
    }

    for (decltype(device_vector)::size_type i = 0; i < 10; ++i) {
        EXPECT_EQ(device_vector[static_cast<vector_size_type>(i)].get(), i + 1);
    }

    // Modify the device vector in different ways, and check that it would work
    // as expected.
    device_vector.clear();
    EXPECT_EQ(device_vector.size(), static_cast<vector_size_type>(0));

    device_vector.emplace_back(10);
    EXPECT_EQ(device_vector.size(), static_cast<vector_size_type>(1));
    EXPECT_EQ(device_vector.at(0).get(), 10);

    device_vector.emplace_back(15);
    EXPECT_EQ(device_vector.size(), static_cast<vector_size_type>(2));
    EXPECT_EQ(device_vector.back().get(), 15);

    device_vector.resize_implicit_unsafe(20);
    for (decltype(device_vector)::size_type i = 0; i < 20; ++i) {
        device_vector.at(i) = 123;
    }
    EXPECT_EQ(device_vector.size(), static_cast<vector_size_type>(20));

    for (nontrivial_class& value : device_vector) {
        EXPECT_EQ(value.get(), 123);
    }

    auto n1 = device_vector.bulk_append_implicit_unsafe(9);
    EXPECT_EQ(device_vector.size(), static_cast<vector_size_type>(29));
    for (decltype(device_vector)::size_type i = 0; i < 9; ++i) {
        device_vector.at(n1 + i) = 123;
    }

    EXPECT_EQ(device_vector.size(), static_cast<vector_size_type>(29));
    for (nontrivial_class& value : device_vector) {
        EXPECT_EQ(value.get(), 123);
    }

    device_vector.resize_implicit_unsafe(40);
    EXPECT_EQ(device_vector.size(), static_cast<vector_size_type>(40));
    device_vector.resize_implicit_unsafe(35);
    EXPECT_EQ(device_vector.size(), static_cast<vector_size_type>(35));
#else
    // This test is meaningless without a good definition of implicit lifetime
    // types.
    GTEST_SKIP()
#endif
}

/// Test(s) for a "resizable" @c vecmem::data::jagged_vector_buffer
TEST_F(core_device_container_test, resizable_jagged_vector_buffer) {

    // Create a buffer with some sufficiently varied capacities.
    vecmem::data::jagged_vector_buffer<int> jagged_buffer(
        std::vector<std::size_t>({0, 16, 10, 15, 8, 3, 0, 0, 55, 2}),
        m_resource, nullptr, vecmem::data::buffer_type::resizable);
    m_copy.setup(jagged_buffer)->wait();

    // Create a device vector on top of the buffer.
    vecmem::jagged_device_vector<int> device_vec(jagged_buffer);

    // Do some tests.
    EXPECT_EQ(device_vec.size(), 10u);

    EXPECT_EQ(device_vec.at(0).size(), 0u);
    EXPECT_EQ(device_vec.at(0).capacity(), 0u);

    EXPECT_EQ(device_vec.at(1).size(), 0u);
    EXPECT_EQ(device_vec.at(1).capacity(), 16u);
    device_vec.at(1).push_back(12);
    device_vec.at(1).push_back(13);
    EXPECT_EQ(device_vec.at(1).size(), 2u);
    EXPECT_EQ(device_vec.at(1).capacity(), 16u);

    EXPECT_EQ(device_vec.at(2).size(), 0u);
    EXPECT_EQ(device_vec.at(2).capacity(), 10u);

    EXPECT_EQ(device_vec.at(3).size(), 0u);
    EXPECT_EQ(device_vec.at(3).capacity(), 15u);

    EXPECT_EQ(device_vec.at(4).size(), 0u);
    EXPECT_EQ(device_vec.at(4).capacity(), 8u);

    EXPECT_EQ(device_vec.at(5).size(), 0u);
    EXPECT_EQ(device_vec.at(5).capacity(), 3u);
    device_vec.at(5).push_back(1);
    device_vec.at(5).push_back(2);
    device_vec.at(5).push_back(3);
    EXPECT_EQ(device_vec.at(5).size(), 3u);
    EXPECT_EQ(device_vec.at(5).capacity(), 3u);

    EXPECT_EQ(device_vec.at(6).size(), 0u);
    EXPECT_EQ(device_vec.at(6).capacity(), 0u);

    EXPECT_EQ(device_vec.at(7).size(), 0u);
    EXPECT_EQ(device_vec.at(7).capacity(), 0u);

    EXPECT_EQ(device_vec.at(8).size(), 0u);
    EXPECT_EQ(device_vec.at(8).capacity(), 55u);
    device_vec.at(8).push_back(54);
    EXPECT_EQ(device_vec.at(8).size(), 1u);
    EXPECT_EQ(device_vec.at(8).capacity(), 55u);

    EXPECT_EQ(device_vec.at(9).size(), 0u);
    EXPECT_EQ(device_vec.at(9).capacity(), 2u);
    device_vec.at(9).push_back(321);
    EXPECT_EQ(device_vec.at(9).size(), 1u);
    EXPECT_EQ(device_vec.at(9).capacity(), 2u);
}

/// Tests with converting between compatible types.
TEST_F(core_device_container_test, conversions) {

    // Create a dummy vector buffer.
    vecmem::data::vector_buffer<int> buffer1d1{
        10, m_resource, vecmem::data::buffer_type::resizable};
    m_copy.setup(buffer1d1)->wait();

    // Check that some conversions compile and work correctly.
    vecmem::data::vector_view<int> view1d1 = buffer1d1;
    vecmem::data::vector_view<int> view1d2;
    view1d2 = view1d1;
    vecmem::data::vector_view<const int> view1d3 = buffer1d1;
    vecmem::data::vector_view<const int> view1d4 = view1d1;
    vecmem::data::vector_view<const int> view1d5;
    view1d5 = view1d1;
    EXPECT_EQ(view1d1, view1d2);
    EXPECT_EQ(view1d1, view1d3);
    EXPECT_EQ(view1d1, view1d4);
    EXPECT_EQ(view1d1, view1d5);
    EXPECT_EQ(view1d2, view1d1);
    EXPECT_EQ(view1d3, view1d1);
    EXPECT_EQ(view1d4, view1d1);
    EXPECT_EQ(view1d5, view1d1);

    // Make copies of the buffer.
    vecmem::data::vector_buffer<int> buffer1d2{std::move(buffer1d1)};
    vecmem::data::vector_buffer<int> buffer1d3;
    buffer1d3 = std::move(buffer1d2);

    // Make sure that the previously created views still point correctly
    // at the "latest" buffer.
    vecmem::data::vector_view<int> view1d6 = buffer1d3;
    EXPECT_EQ(view1d6, view1d1);

    // Create a dummy jagged vector buffer.
    vecmem::data::jagged_vector_buffer<int> buffer2d1(
        std::vector<std::size_t>({0, 16, 10, 15, 8, 3, 0, 0, 55, 2}),
        m_resource, nullptr, vecmem::data::buffer_type::resizable);
    m_copy.setup(buffer2d1)->wait();

    // Check that some conversions compile and work correctly.
    vecmem::data::jagged_vector_view<int> view2d1 = buffer2d1;
    vecmem::data::jagged_vector_view<int> view2d2;
    view2d2 = view2d1;
    vecmem::data::jagged_vector_view<const int> view2d3 = buffer2d1;
    vecmem::data::jagged_vector_view<const int> view2d4 = view2d1;
    vecmem::data::jagged_vector_view<const int> view2d5;
    view2d5 = view2d1;
    EXPECT_EQ(view2d1, view2d2);
    EXPECT_EQ(view2d1, view2d3);
    EXPECT_EQ(view2d1, view2d4);
    EXPECT_EQ(view2d1, view2d5);
    EXPECT_EQ(view2d2, view2d1);
    EXPECT_EQ(view2d3, view2d1);
    EXPECT_EQ(view2d4, view2d1);
    EXPECT_EQ(view2d5, view2d1);

    // Make copies of the buffer.
    vecmem::data::jagged_vector_buffer<int> buffer2d2{std::move(buffer2d1)};
    vecmem::data::jagged_vector_buffer<int> buffer2d3;
    buffer2d3 = std::move(buffer2d2);

    // Make sure that the previously created views still point correctly
    // at the "latest" buffer.
    vecmem::data::jagged_vector_view<int> view2d6 = buffer2d3;
    EXPECT_EQ(view2d6, view2d1);

    // Create a dummy jagged vector.
    vecmem::jagged_vector<int> vector2d1 = {{
                                                {{1, 2, 3, 4}, &m_resource},
                                                {{5, 6}, &m_resource},
                                                {{7}, &m_resource},
                                                {{8, 9}, &m_resource},
                                                {{10}, &m_resource},
                                            },
                                            &m_resource};

    // Made a data object out of it.
    vecmem::data::jagged_vector_data<int> data2d1 = vecmem::get_data(vector2d1);
    vecmem::data::jagged_vector_view<int> view2d7 = data2d1;

    // Make copies of the data.
    vecmem::data::jagged_vector_data<int> data2d2{std::move(data2d1)};
    vecmem::data::jagged_vector_data<int> data2d3;
    data2d3 = std::move(data2d2);

    // Make sure that the "latest" data object is still the same as the first
    // one.
    vecmem::data::jagged_vector_view<int> view2d8 = data2d3;
    EXPECT_EQ(view2d7, view2d8);
}

TEST_F(core_device_container_test, vector_iteration) {

    // Set up a simple host vector.
    vecmem::vector<int> host_vector{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                                    &m_resource};

    // Create a non-const device vector from it.
    vecmem::device_vector<int> device_vector1(vecmem::get_data(host_vector));

    // Check that non-const iteration works.
    vecmem::device_vector<int>::iterator it1 = device_vector1.begin();
    vecmem::device_vector<int>::iterator end1 = device_vector1.end();
    for (int i = 1; it1 != end1; ++it1, ++i) {
        EXPECT_EQ(*it1, i);
    }

    // Check that const iteration works.
    vecmem::device_vector<int>::const_iterator it2 = device_vector1.cbegin();
    vecmem::device_vector<int>::const_iterator end2 = device_vector1.cend();
    for (int i = 1; it2 != end2; ++it2, ++i) {
        EXPECT_EQ(*it2, i);
    }
    it2 = [](const auto& vec) { return vec.begin(); }(device_vector1);
    end2 = [](const auto& vec) { return vec.end(); }(device_vector1);
    for (int i = 1; it2 != end2; ++it2, ++i) {
        EXPECT_EQ(*it2, i);
    }

    // Create a const device vector from it.
    vecmem::device_vector<const int> device_vector2(
        vecmem::get_data(host_vector));

    // Check that "non-const" iteration works.
    vecmem::device_vector<const int>::iterator it3 = device_vector2.begin();
    vecmem::device_vector<const int>::iterator end3 = device_vector2.end();
    for (int i = 1; it3 != end3; ++it3, ++i) {
        EXPECT_EQ(*it3, i);
    }

    // Check that const iteration works.
    vecmem::device_vector<const int>::const_iterator it4 =
        device_vector2.cbegin();
    vecmem::device_vector<const int>::const_iterator end4 =
        device_vector2.cend();
    for (int i = 1; it4 != end4; ++it4, ++i) {
        EXPECT_EQ(*it4, i);
    }
    it4 = [](const auto& vec) { return vec.begin(); }(device_vector2);
    end4 = [](const auto& vec) { return vec.end(); }(device_vector2);
    for (int i = 1; it4 != end4; ++it4, ++i) {
        EXPECT_EQ(*it4, i);
    }
}

TEST_F(core_device_container_test, jagged_vector_iteration) {

    // Set up a "simple" host vector.
    vecmem::jagged_vector<int> host_vector = {
        {
            {{1, 2, 3, 4}, &m_resource},
            {{5, 6}, &m_resource},
            vecmem::vector<int>{&m_resource},
            {{7, 8, 9}, &m_resource},
            {{10}, &m_resource},
        },
        &m_resource};
    auto host_data = vecmem::get_data(host_vector);

    // Create a non-const device vector from it.
    vecmem::jagged_device_vector<int> device_vector1(
        vecmem::get_data(host_data));

    // Check that non-const iteration works.
    int i = 1;
    vecmem::jagged_device_vector<int>::iterator it1 = device_vector1.begin();
    vecmem::jagged_device_vector<int>::iterator end1 = device_vector1.end();
    for (; it1 != end1; ++it1) {
        for (auto it = it1->begin(); it != it1->end(); ++it, ++i) {
            EXPECT_EQ(*it, i);
        }
    }

    // Check that const iteration works.
    i = 1;
    vecmem::jagged_device_vector<int>::const_iterator it2 =
        device_vector1.cbegin();
    vecmem::jagged_device_vector<int>::const_iterator end2 =
        device_vector1.cend();
    for (; it2 != end2; ++it2) {
        for (auto it = it2->cbegin(); it != it2->cend(); ++it, ++i) {
            EXPECT_EQ(*it, i);
        }
    }
    i = 1;
    it2 = [](const auto& vec) { return vec.begin(); }(device_vector1);
    end2 = [](const auto& vec) { return vec.end(); }(device_vector1);
    for (; it2 != end2; ++it2) {
        for (auto it = it2->cbegin(); it != it2->cend(); ++it, ++i) {
            EXPECT_EQ(*it, i);
        }
    }

    // Create a const device vector from it.
    vecmem::jagged_device_vector<const int> device_vector2(
        vecmem::get_data(host_data));

    // Check that "non-const" iteration works.
    i = 1;
    vecmem::jagged_device_vector<const int>::iterator it3 =
        device_vector2.begin();
    vecmem::jagged_device_vector<const int>::iterator end3 =
        device_vector2.end();
    for (; it3 != end3; ++it3) {
        for (auto it = it3->begin(); it != it3->end(); ++it, ++i) {
            EXPECT_EQ(*it, i);
        }
    }

    // Check that const iteration works.
    i = 1;
    vecmem::jagged_device_vector<const int>::const_iterator it4 =
        device_vector2.cbegin();
    vecmem::jagged_device_vector<const int>::const_iterator end4 =
        device_vector2.cend();
    for (; it4 != end4; ++it4) {
        for (auto it = it4->cbegin(); it != it4->cend(); ++it, ++i) {
            EXPECT_EQ(*it, i);
        }
    }
    i = 1;
    it4 = [](const auto& vec) { return vec.begin(); }(device_vector2);
    end4 = [](const auto& vec) { return vec.end(); }(device_vector2);
    for (; it4 != end4; ++it4) {
        for (auto it = it4->cbegin(); it != it4->cend(); ++it, ++i) {
            EXPECT_EQ(*it, i);
        }
    }
}
