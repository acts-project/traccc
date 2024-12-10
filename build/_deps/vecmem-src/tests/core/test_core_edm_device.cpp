/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/edm/device.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <vector>

namespace {

/// Dummy container interface for the test
template <typename BASE>
struct interface {};

}  // namespace

TEST(core_edm_device_test, construct) {

    using schema =
        vecmem::edm::schema<vecmem::edm::type::scalar<int>,
                            vecmem::edm::type::vector<float>,
                            vecmem::edm::type::jagged_vector<double>>;
    using const_schema = vecmem::edm::details::add_const_t<schema>;

    vecmem::edm::view<schema> view1{};
    vecmem::edm::view<const_schema> view2{};

    vecmem::edm::device<schema, interface> device1{view1};
    vecmem::edm::device<const_schema, interface> device2{view1};
    vecmem::edm::device<const_schema, interface> device3{view2};
}

TEST(core_edm_device_test, members) {

    int value1 = 1;
    std::vector<float> value2{2.0f, 3.0f};

    using schema = vecmem::edm::schema<vecmem::edm::type::scalar<int>,
                                       vecmem::edm::type::vector<float>>;
    using const_schema = vecmem::edm::details::add_const_t<schema>;

    vecmem::edm::view<schema> view{2};  // 2 is the size of the vector variable,
                                        // not the number of variables
    view.get<0>() = &value1;
    view.get<1>() = {static_cast<unsigned int>(value2.size()), value2.data()};

    vecmem::edm::device<schema, interface> device1{view};
    vecmem::edm::device<const_schema, interface> device2{view};

    EXPECT_EQ(device2.get<0>(), value1);
    ASSERT_EQ(device2.get<1>().size(), value2.size());
    EXPECT_EQ(device2.get<1>()[0], value2[0]);
    EXPECT_EQ(device2.get<1>()[1], value2[1]);

    device1.get<0>() = 2;
    device1.get<1>()[0] = 4.0f;
    device1.get<1>()[1] = 5.0f;

    EXPECT_EQ(device2.get<0>(), 2);
    EXPECT_EQ(device2.get<1>()[0], 4.0f);
    EXPECT_EQ(device2.get<1>()[1], 5.0f);
}
