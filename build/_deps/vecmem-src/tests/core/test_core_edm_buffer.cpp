/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/edm/buffer.hpp"
#include "vecmem/edm/device.hpp"
#include "vecmem/memory/host_memory_resource.hpp"
#include "vecmem/utils/copy.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <algorithm>
#include <numeric>
#include <vector>

/// Test case for @c vecmem::edm::buffer.
class core_edm_buffer_test : public testing::Test {

protected:
    /// Schema without any jagged vectors.
    using simple_schema = vecmem::edm::schema<
        vecmem::edm::type::scalar<int>, vecmem::edm::type::vector<float>,
        vecmem::edm::type::scalar<double>, vecmem::edm::type::vector<double>,
        vecmem::edm::type::vector<int>>;
    /// Constant schema without any jagged vectors.
    using simple_const_schema =
        vecmem::edm::details::add_const_t<simple_schema>;

    /// Schema with some jagged vectors.
    using jagged_schema =
        vecmem::edm::schema<vecmem::edm::type::vector<float>,
                            vecmem::edm::type::jagged_vector<double>,
                            vecmem::edm::type::scalar<int>,
                            vecmem::edm::type::jagged_vector<int>>;
    /// Constant schema with some jagged vectors.
    using jagged_const_schema =
        vecmem::edm::details::add_const_t<jagged_schema>;

    /// Dummy container interface for the test
    template <typename BASE>
    struct interface {};

    /// Memory resource for the test(s)
    vecmem::host_memory_resource m_resource;
    /// Copy object for the test(s)
    vecmem::copy m_copy;

};  // class core_edm_buffer_test

/// Capacities used for the test(s)
static const std::vector<unsigned int> CAPACITIES = {10, 100, 1000};
static const unsigned int CAPACITY =
    static_cast<unsigned int>(CAPACITIES.size());

TEST_F(core_edm_buffer_test, construct) {

    // Test the creation of fixed sized and resizable "simple buffers".
    vecmem::edm::buffer<simple_schema> buffer1{
        CAPACITY, m_resource, vecmem::data::buffer_type::fixed_size};
    vecmem::edm::buffer<simple_schema> buffer2{
        CAPACITY, m_resource, vecmem::data::buffer_type::resizable};

    // Test the creation of fixed sized and resizable "jagged buffers".
    vecmem::edm::buffer<jagged_schema> buffer3{
        CAPACITIES, m_resource, nullptr, vecmem::data::buffer_type::fixed_size};
    vecmem::edm::buffer<jagged_schema> buffer4{
        CAPACITIES, m_resource, nullptr, vecmem::data::buffer_type::resizable};
}

TEST_F(core_edm_buffer_test, get_data) {

    // Lambda creating constant views.
    auto create_const_view = [](const auto& buffer) {
        return vecmem::get_data(buffer);
    };

    // Construct a fixed sized, simple buffer.
    vecmem::edm::buffer<simple_schema> buffer1{
        CAPACITY, m_resource, vecmem::data::buffer_type::fixed_size};

    // Make views of it.
    vecmem::edm::view<simple_schema> view1 = vecmem::get_data(buffer1);
    vecmem::edm::view<simple_const_schema> view2 = vecmem::get_data(buffer1);
    vecmem::edm::view<simple_const_schema> view3 = create_const_view(buffer1);

    // Lambda checking the simple, fixed sized views.
    auto check_simple_fixed = [&](const auto& view) {
        EXPECT_EQ(view.capacity(), CAPACITY);
        EXPECT_EQ(view.size().size(), 0u);
        EXPECT_EQ(view.size().ptr(), nullptr);
        auto check_scalar = [](const auto& v) { EXPECT_NE(v, nullptr); };
        auto check_vector = [&](const auto& v) {
            EXPECT_EQ(v.size(), CAPACITY);
            EXPECT_EQ(v.size_ptr(), nullptr);
            EXPECT_EQ(v.capacity(), CAPACITY);
            EXPECT_NE(v.ptr(), nullptr);
        };
        check_scalar(view.template get<0>());
        check_vector(view.template get<1>());
        check_scalar(view.template get<2>());
        check_vector(view.template get<3>());
        check_vector(view.template get<4>());
    };

    // Check the views.
    check_simple_fixed(view1);
    check_simple_fixed(view2);
    check_simple_fixed(view3);

    // Construct a resizable, simple buffer.
    vecmem::edm::buffer<simple_schema> buffer2{
        CAPACITY, m_resource, vecmem::data::buffer_type::resizable};
    m_copy.memset(buffer2.size(), 0)->wait();

    // Make views of it.
    vecmem::edm::view<simple_schema> view4 = vecmem::get_data(buffer2);
    vecmem::edm::view<simple_const_schema> view5 = vecmem::get_data(buffer2);
    vecmem::edm::view<simple_const_schema> view6 = create_const_view(buffer2);

    // Lambda checking the simple, resizable views.
    auto check_simple_resizable = [&](const auto& view) {
        EXPECT_EQ(view.capacity(), CAPACITY);
        EXPECT_EQ(view.size().size(), sizeof(unsigned int));
        EXPECT_NE(view.size().ptr(), nullptr);
        auto check_scalar = [](const auto& v) { EXPECT_NE(v, nullptr); };
        auto check_vector = [&](const auto& v) {
            EXPECT_EQ(v.size(), 0u);
            EXPECT_EQ(static_cast<const void*>(v.size_ptr()),
                      static_cast<const void*>(view.size().ptr()));
            EXPECT_EQ(v.capacity(), CAPACITY);
            EXPECT_NE(v.ptr(), nullptr);
        };
        check_scalar(view.template get<0>());
        check_vector(view.template get<1>());
        check_scalar(view.template get<2>());
        check_vector(view.template get<3>());
        check_vector(view.template get<4>());
    };

    // Check the views.
    check_simple_resizable(view4);
    check_simple_resizable(view5);
    check_simple_resizable(view6);

    // Construct a fixed sized, jagged buffer.
    vecmem::edm::buffer<jagged_schema> buffer3{
        CAPACITIES, m_resource, nullptr, vecmem::data::buffer_type::fixed_size};

    // Make views of it.
    vecmem::edm::view<jagged_schema> view7 = vecmem::get_data(buffer3);
    vecmem::edm::view<jagged_const_schema> view8 = vecmem::get_data(buffer3);
    vecmem::edm::view<jagged_const_schema> view9 = create_const_view(buffer3);

    // Lambda checking the jagged, fixed sized views.
    auto check_jagged_fixed = [&](const auto& view) {
        EXPECT_EQ(view.capacity(), CAPACITY);
        EXPECT_EQ(view.size().size(), 0u);
        EXPECT_EQ(view.size().ptr(), nullptr);
        auto check_scalar = [](const auto& v) { EXPECT_NE(v, nullptr); };
        auto check_vector = [&](const auto& v) {
            EXPECT_EQ(v.size(), CAPACITY);
            EXPECT_EQ(v.size_ptr(), nullptr);
            EXPECT_EQ(v.capacity(), CAPACITY);
            EXPECT_NE(v.ptr(), nullptr);
        };
        auto check_jagged = [&](const auto& v) {
            EXPECT_EQ(v.size(), CAPACITY);
            EXPECT_EQ(v.capacity(), CAPACITY);
            EXPECT_NE(v.ptr(), nullptr);
            EXPECT_NE(v.host_ptr(), nullptr);
            EXPECT_EQ(v.host_ptr(), v.ptr());
            for (unsigned int i = 0; i < CAPACITY; ++i) {
                EXPECT_EQ(v.host_ptr()[i].size(), CAPACITIES[i]);
                EXPECT_EQ(v.host_ptr()[i].size_ptr(), nullptr);
                EXPECT_EQ(v.host_ptr()[i].capacity(), CAPACITIES[i]);
                EXPECT_NE(v.host_ptr()[i].ptr(), nullptr);
            }
        };
        check_vector(view.template get<0>());
        check_jagged(view.template get<1>());
        check_scalar(view.template get<2>());
        check_jagged(view.template get<3>());
    };

    // Check the views.
    check_jagged_fixed(view7);
    check_jagged_fixed(view8);
    check_jagged_fixed(view9);

    // Construct a resizable, jagged buffer.
    vecmem::edm::buffer<jagged_schema> buffer4{
        CAPACITIES, m_resource, nullptr, vecmem::data::buffer_type::resizable};
    m_copy.memset(buffer4.size(), 0)->wait();

    // Make views of it.
    vecmem::edm::view<jagged_schema> view10 = vecmem::get_data(buffer4);
    vecmem::edm::view<jagged_const_schema> view11 = vecmem::get_data(buffer4);
    vecmem::edm::view<jagged_const_schema> view12 = create_const_view(buffer4);

    // Lambda checking the jagged, resizable views.
    auto check_jagged_resizable = [&](const auto& view) {
        EXPECT_EQ(view.capacity(), CAPACITY);
        // There are 2 jagged vectors in the container, which all need 3
        // unsigned integers for their sizes.
        EXPECT_EQ(view.size().size(), 2 * CAPACITY * sizeof(unsigned int));
        EXPECT_NE(view.size().ptr(), nullptr);
        auto check_scalar = [](const auto& v) { EXPECT_NE(v, nullptr); };
        auto check_vector = [&](const auto& v) {
            EXPECT_EQ(v.size(), CAPACITY);
            EXPECT_EQ(v.size_ptr(), nullptr);
            EXPECT_EQ(v.capacity(), CAPACITY);
            EXPECT_NE(v.ptr(), nullptr);
        };
        auto check_jagged = [&](const auto& v) {
            EXPECT_EQ(v.size(), CAPACITY);
            EXPECT_EQ(v.capacity(), CAPACITY);
            EXPECT_NE(v.ptr(), nullptr);
            EXPECT_NE(v.host_ptr(), nullptr);
            EXPECT_EQ(v.host_ptr(), v.ptr());
            for (unsigned int i = 0; i < CAPACITY; ++i) {
                EXPECT_EQ(v.host_ptr()[i].size(), 0u);
                // Checking the exact value of v.host_ptr()[i].size_ptr()
                // would be a bit too hard, to be worth it...
                EXPECT_EQ(v.host_ptr()[i].capacity(), CAPACITIES[i]);
                EXPECT_NE(v.host_ptr()[i].ptr(), nullptr);
            }
        };
        check_vector(view.template get<0>());
        check_jagged(view.template get<1>());
        check_scalar(view.template get<2>());
        check_jagged(view.template get<3>());
    };

    // Check the views.
    check_jagged_resizable(view10);
    check_jagged_resizable(view11);
    check_jagged_resizable(view12);
}

TEST_F(core_edm_buffer_test, device) {

    // Construct a fixed sized, simple buffer.
    vecmem::edm::buffer<simple_schema> buffer1{
        CAPACITY, m_resource, vecmem::data::buffer_type::fixed_size};

    // Make a device container on top of it.
    vecmem::edm::device<simple_schema, interface> device1{buffer1};
    ASSERT_EQ(device1.size(), CAPACITY);
    ASSERT_EQ(device1.capacity(), CAPACITY);
    auto check_fixed_vector = [&](const auto& v) {
        ASSERT_EQ(v.size(), CAPACITY);
        ASSERT_EQ(v.capacity(), CAPACITY);
    };
    check_fixed_vector(device1.get<1>());
    check_fixed_vector(device1.get<3>());
    check_fixed_vector(device1.get<4>());

    // Fill it in some non-trivial way.
    device1.get<0>() = 1;
    std::fill(device1.get<1>().begin(), device1.get<1>().end(), 2.f);
    device1.get<2>() = 3.;
    std::fill(device1.get<3>().begin(), device1.get<3>().end(), 4.);
    std::iota(device1.get<4>().begin(), device1.get<4>().end(), 5);

    // Check the values. Making sure that there is no overlap between the
    // variables in the buffer due to some bug.
    EXPECT_EQ(device1.get<0>(), 1);
    EXPECT_DOUBLE_EQ(device1.get<2>(), 3.);
    std::for_each(device1.get<1>().begin(), device1.get<1>().end(),
                  [](const auto& v) { EXPECT_FLOAT_EQ(v, 2.f); });
    std::for_each(device1.get<3>().begin(), device1.get<3>().end(),
                  [](const auto& v) { EXPECT_DOUBLE_EQ(v, 4.); });
    for (unsigned int i = 0; i < CAPACITY; ++i) {
        EXPECT_EQ(device1.get<4>()[i], static_cast<int>(5 + i));
    }

    // Construct a resizable, simple buffer.
    vecmem::edm::buffer<simple_schema> buffer2{
        CAPACITY, m_resource, vecmem::data::buffer_type::resizable};
    m_copy.memset(buffer2.size(), 0)->wait();

    // Make a device container on top of it.
    vecmem::edm::device<simple_schema, interface> device2{buffer2};
    ASSERT_EQ(device2.size(), 0u);
    ASSERT_EQ(device2.capacity(), CAPACITY);
    auto check_resizable_vector = [&](const auto& v) {
        ASSERT_EQ(v.size(), 0u);
        ASSERT_EQ(v.capacity(), CAPACITY);
    };
    check_resizable_vector(device2.get<1>());
    check_resizable_vector(device2.get<3>());
    check_resizable_vector(device2.get<4>());

    // Fill it in some non-trivial way.
    device2.get<0>() = 1;
    device2.get<2>() = 2.;
    for (unsigned int i = 0; i < 2; ++i) {
        const unsigned int ii = device2.push_back_default();
        EXPECT_EQ(ii, i);
        device2.get<1>()[ii] = 3.f;
        device2.get<3>()[ii] = 4.;
        device2.get<4>()[ii] = static_cast<int>(5 + i);
    }

    // Check the values.
    EXPECT_EQ(device2.get<0>(), 1);
    EXPECT_DOUBLE_EQ(device2.get<2>(), 2.);
    EXPECT_EQ(device2.size(), 2u);
    EXPECT_EQ(device2.get<1>().size(), 2u);
    EXPECT_EQ(device2.get<3>().size(), 2u);
    ASSERT_EQ(device2.get<4>().size(), 2u);
    std::for_each(device2.get<1>().begin(), device2.get<1>().end(),
                  [](const auto& v) { EXPECT_FLOAT_EQ(v, 3.f); });
    std::for_each(device2.get<3>().begin(), device2.get<3>().end(),
                  [](const auto& v) { EXPECT_DOUBLE_EQ(v, 4.); });
    for (unsigned int i = 0; i < 2u; ++i) {
        EXPECT_EQ(device2.get<4>()[i], static_cast<int>(5 + i));
    }

    // Construct a fixed sized, jagged buffer.
    vecmem::edm::buffer<jagged_schema> buffer3{
        CAPACITIES, m_resource, nullptr, vecmem::data::buffer_type::fixed_size};

    // Make a device container on top of it.
    vecmem::edm::device<jagged_schema, interface> device3{buffer3};
    ASSERT_EQ(device3.size(), CAPACITY);
    ASSERT_EQ(device3.capacity(), CAPACITY);
    auto check_fixed_jagged = [&](const auto& v) {
        ASSERT_EQ(v.size(), CAPACITY);
        ASSERT_EQ(v.capacity(), CAPACITY);
        for (std::size_t i = 0; i < CAPACITY; ++i) {
            ASSERT_EQ(v[i].size(), CAPACITIES[i]);
            ASSERT_EQ(v[i].capacity(), CAPACITIES[i]);
        }
    };
    check_fixed_vector(device3.get<0>());
    check_fixed_jagged(device3.get<1>());
    check_fixed_jagged(device3.get<3>());

    // Fill it in some non-trivial way.
    std::fill(device3.get<0>().begin(), device3.get<0>().end(), 1.f);
    device3.get<2>() = 2;
    for (unsigned int i = 0; i < CAPACITY; ++i) {
        std::fill(device3.get<1>()[i].begin(), device3.get<1>()[i].end(), 3.);
        std::iota(device3.get<3>()[i].begin(), device3.get<3>()[i].end(), 4);
    }

    // Check the values.
    std::for_each(device3.get<0>().begin(), device3.get<0>().end(),
                  [](const auto& v) { EXPECT_FLOAT_EQ(v, 1.f); });
    EXPECT_EQ(device3.get<2>(), 2);
    for (unsigned int i = 0; i < CAPACITY; ++i) {
        std::for_each(device3.get<1>()[i].begin(), device3.get<1>()[i].end(),
                      [](const auto& v) { EXPECT_DOUBLE_EQ(v, 3.); });
        for (unsigned int j = 0; j < device3.get<3>()[i].size(); ++j) {
            EXPECT_EQ(device3.get<3>()[i][j], static_cast<int>(4 + j));
        }
    }

    // Construct a resizable, jagged buffer.
    vecmem::edm::buffer<jagged_schema> buffer4{
        CAPACITIES, m_resource, nullptr, vecmem::data::buffer_type::resizable};
    m_copy.memset(buffer4.size(), 0)->wait();

    // Make a device container on top of it.
    vecmem::edm::device<jagged_schema, interface> device4{buffer4};
    ASSERT_EQ(device4.size(), CAPACITY);
    ASSERT_EQ(device4.capacity(), CAPACITY);
    auto check_resizable_jagged = [&](const auto& v) {
        ASSERT_EQ(v.size(), CAPACITY);
        ASSERT_EQ(v.capacity(), CAPACITY);
        for (std::size_t i = 0; i < CAPACITY; ++i) {
            ASSERT_EQ(v[i].size(), 0u);
            ASSERT_EQ(v[i].capacity(), CAPACITIES[i]);
        }
    };
    check_fixed_vector(device4.get<0>());
    check_resizable_jagged(device4.get<1>());
    check_resizable_jagged(device4.get<3>());

    // Fill it in some non-trivial way.
    std::fill(device4.get<0>().begin(), device4.get<0>().end(), 6.f);
    device4.get<2>() = 7;
    std::for_each(device4.get<1>().begin(), device4.get<1>().end(),
                  [&](auto v) {
                      const unsigned int size = v.capacity() / 2u;
                      for (unsigned int i = 0; i < size; ++i) {
                          v.push_back(8. + static_cast<double>(i));
                      }
                  });
    std::for_each(device4.get<3>().begin(), device4.get<3>().end(),
                  [&](auto v) {
                      const unsigned int size = v.capacity() / 2u;
                      for (unsigned int i = 0; i < size; ++i) {
                          v.push_back(9 + static_cast<int>(i));
                      }
                  });

    // Check the values.
    std::for_each(device4.get<0>().begin(), device4.get<0>().end(),
                  [](const auto& v) { EXPECT_FLOAT_EQ(v, 6.f); });
    EXPECT_EQ(device4.get<2>(), 7);
    for (unsigned int i = 0; i < CAPACITY; ++i) {
        EXPECT_EQ(device4.get<1>()[i].size(), CAPACITIES[i] / 2u);
        for (unsigned int j = 0; j < device4.get<1>()[i].size(); ++j) {
            EXPECT_DOUBLE_EQ(device4.get<1>()[i][j],
                             8. + static_cast<double>(j));
        }
        EXPECT_EQ(device4.get<3>()[i].size(), CAPACITIES[i] / 2u);
        for (unsigned int j = 0; j < device4.get<3>()[i].size(); ++j) {
            EXPECT_EQ(device4.get<3>()[i][j], 9 + static_cast<int>(j));
        }
    }
}
