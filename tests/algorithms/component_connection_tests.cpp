/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#include "edm/cell.hpp"
#include "edm/cluster.hpp"
#include "cpu/include/edm/cell_container.hpp"
#include "cpu/include/algorithms/component_connection.hpp"

#include <gtest/gtest.h>

// This defines the local frame test suite
TEST(algorithms, component_connection){

    /// Following [DOI: 10.1109/DASIP48288.2019.9049184]
    std::vector<traccc::cell> cell_items = 
        { {1, 0, 1., 0. }, 
          {8, 4, 2., 0.}, 
          {10, 4, 3., 0.}, 
          {9, 5, 4., 0.}, 
          {10, 5, 5., 0}, 
          {12, 12, 6, 0}, 
          {3, 13, 7, 0}, 
          {11, 13, 8, 0}, 
          {4, 14, 9, 0 } };

    traccc::cell_collection cells;
    cells.items = cell_items;
    cells.modcfg.module = 0;

    traccc::component_connection ccl;
    auto clusters = ccl(cells);
   
    ASSERT_EQ(clusters.items.size(), 4u);

}

// Google Test can be run manually from the main() function
// or, it can be linked to the gtest_main library for an already
// set-up main() function primed to accept Google Test test cases.
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
