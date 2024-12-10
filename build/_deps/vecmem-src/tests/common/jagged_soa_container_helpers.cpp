/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "jagged_soa_container_helpers.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace vecmem::testing {

/// The "outer size" of the created containers and buffers.
static const unsigned int SIZE = 10;

void fill(jagged_soa_container::host& obj) {

    obj.resize(SIZE);
    obj.count() = 55;
    obj.average() = 3.141592f;
    for (std::size_t i = 0; i < obj.size(); ++i) {
        obj.measurement()[i] = 1.0f * static_cast<float>(i);
        obj.index()[i] = static_cast<int>(i);
        obj.measurements()[i].resize(5 + i);
        for (std::size_t j = 0; j < obj.measurements()[i].size(); ++j) {
            obj.measurements()[i][j] = 1.0f * static_cast<float>(i + j);
        }
        obj.indices()[i].resize(5 + i);
        for (std::size_t j = 0; j < obj.indices()[i].size(); ++j) {
            obj[i].indices()[j] = static_cast<int>(i + j);
        }
    }
}

void compare(const jagged_soa_container::const_view& view1,
             const jagged_soa_container::const_view& view2) {

    // Create device containers on top of the views.
    const jagged_soa_container::const_device device1{view1};
    const jagged_soa_container::const_device device2{view2};

    // Check the size of the containers.
    EXPECT_EQ(device1.size(), device2.size());

    // Compare the scalar variables.
    EXPECT_EQ(device1.count(), device2.count());
    EXPECT_FLOAT_EQ(device1.average(), device2.average());

    // Compare the vector variables.
    auto compare_vector = [](const auto& lhs, const auto& rhs) {
        ASSERT_EQ(lhs.size(), rhs.size());
        for (unsigned int i = 0; i < lhs.size(); ++i) {
            EXPECT_EQ(lhs[i], rhs[i]);
        }
    };
    compare_vector(device1.measurement(), device2.measurement());
    compare_vector(device1.index(), device2.index());

    // Compare the jagged vector variables.
    auto compare_jagged = [&](const auto& lhs, const auto& rhs) {
        ASSERT_EQ(lhs.size(), rhs.size());
        for (unsigned int i = 0; i < lhs.size(); ++i) {
            compare_vector(lhs[i], rhs[i]);
        }
    };
    compare_jagged(device1.measurements(), device2.measurements());
    compare_jagged(device1.indices(), device2.indices());
}

void make_buffer(jagged_soa_container::buffer& buffer, memory_resource& main_mr,
                 memory_resource& host_mr, data::buffer_type buffer_type) {

    std::vector<unsigned int> sizes(SIZE);
    switch (buffer_type) {
        case data::buffer_type::fixed_size:
            std::iota(sizes.begin(), sizes.end(), 5);
            buffer = {sizes, main_mr, &host_mr, buffer_type};
            break;
        case data::buffer_type::resizable:
            std::fill(sizes.begin(), sizes.end(), 20);
            buffer = {sizes, main_mr, &host_mr, buffer_type};
            break;
        default:
            throw std::runtime_error("Unknown buffer type received");
    }
}

}  // namespace vecmem::testing
