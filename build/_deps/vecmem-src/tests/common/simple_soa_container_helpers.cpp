/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "simple_soa_container_helpers.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <type_traits>

namespace vecmem::testing {

void fill(simple_soa_container::host& obj) {

    obj.resize(10);
    obj.count() = 55;
    obj.average() = 3.141592f;
    for (std::size_t i = 0; i < obj.size(); ++i) {
        obj.measurement()[i] = 1.0f * static_cast<float>(i);
        obj.index()[i] = static_cast<int>(i);
    }
}

void compare(const simple_soa_container::const_view& view1,
             const simple_soa_container::const_view& view2) {

    // Create device containers on top of the views.
    const simple_soa_container::const_device device1{view1};
    const simple_soa_container::const_device device2{view2};

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
}

void make_buffer(simple_soa_container::buffer& buffer, memory_resource& main_mr,
                 memory_resource&, data::buffer_type buffer_type) {

    switch (buffer_type) {
        case data::buffer_type::fixed_size:
            buffer = {10, main_mr, buffer_type};
            break;
        case data::buffer_type::resizable:
            buffer = {20, main_mr, buffer_type};
            break;
        default:
            throw std::runtime_error("Unsupported buffer type");
    }
}

}  // namespace vecmem::testing
