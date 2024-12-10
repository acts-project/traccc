/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/edm/view.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <vector>

TEST(core_edm_view_test, construct_assign) {

    vecmem::edm::view<vecmem::edm::schema<
        vecmem::edm::type::scalar<int>, vecmem::edm::type::vector<float>,
        vecmem::edm::type::jagged_vector<double>>>
        view1{};
    vecmem::edm::view<
        vecmem::edm::schema<vecmem::edm::type::scalar<const int>,
                            vecmem::edm::type::vector<const float>,
                            vecmem::edm::type::jagged_vector<const double>>>
        view2{view1};
    vecmem::edm::view<
        vecmem::edm::schema<vecmem::edm::type::scalar<const int>,
                            vecmem::edm::type::vector<const float>,
                            vecmem::edm::type::jagged_vector<const double>>>
        view3{};
    view3 = view1;
}

TEST(core_edm_view_test, members) {

    int value1 = 1;
    std::vector<float> value2{2.0f, 3.0f};

    vecmem::edm::view<vecmem::edm::schema<vecmem::edm::type::scalar<int>,
                                          vecmem::edm::type::vector<float>>>
        view{2};
    view.get<0>() = &value1;
    view.get<1>() = {static_cast<unsigned int>(value2.size()), value2.data()};

    EXPECT_EQ(view.get<0>(), &value1);
    EXPECT_EQ(view.get<1>().size(), value2.size());
    EXPECT_EQ(view.get<1>().ptr(), value2.data());
}
