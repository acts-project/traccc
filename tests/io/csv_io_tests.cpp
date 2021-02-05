/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#include "csv/csv_io.hpp"

#include <gtest/gtest.h>

// This defines the local frame test suite
TEST(io, csv_read_single_module)
{
    std::string file = "/scratch/asalzbur/dev/traccc/data/single_module/cells.csv";

    // module_id,channel0,channel1,timestamp,value
    // 0,174,880,0,0.23
    // 0,174,879,0,0.22
    // 0,123,32,0,0.002
    // 0,123,33,0,0.12
    // 0,123,34,0,0.17
    // 0,124,34,0,0.08

    traccc::cell_reader creader(file, {"cannel0","channel1","activation","time"} );

    auto single_module_cells = traccc::read_cells_per_module(creader);

    ASSERT_EQ(single_module_cells.module_id, 0u);
    auto expected_range0 = std::array<traccc::channel_id,2>{123u,174u};
    ASSERT_EQ(single_module_cells.range0, expected_range0);
    auto expected_range1 = std::array<traccc::channel_id,2>{32u,880u};
    ASSERT_EQ(single_module_cells.range1, expected_range1);
    ASSERT_EQ(single_module_cells.items.size(), 6u);

}


// This defines the local frame test suite
TEST(io, csv_read_two_modules)
{
    std::string file = "/scratch/asalzbur/dev/traccc/data/two_modules/cells.csv";

    // module_id,channel0,channel1,timestamp,value
    // 0,174,880,0,0.23
    // 0,174,879,0,0.22
    // 0,123,32,0,0.002
    // 0,123,33,0,0.12
    // 0,123,34,0,0.17
    // 0,124,34,0,0.08
    // 1,0,0,0,0. - triggereing a new module 
    // 1,0,4,0,0.17
    // 1,0,5,0,0.37
    // 1,0,6,0,0.27
    // 1,1,6,0,0.07
    // 1,5,98,0,0.15
    // 1,22,8,0,0.14
    // 1,22,9,0,0.14
    // 1,22,10,0,0.14

    traccc::cell_reader creader(file, {"module_id", "cannel0","channel1","activation","time"} );
    auto first_module_cells = traccc::read_cells_per_module(creader);

    ASSERT_EQ(first_module_cells.items.size(), 6u);

    ASSERT_EQ(first_module_cells.module_id, 0u);
    auto expected_range0 = std::array<traccc::channel_id,2>{123u,174u};
    ASSERT_EQ(first_module_cells.range0, expected_range0);
    auto expected_range1 = std::array<traccc::channel_id,2>{32u,880u};
    ASSERT_EQ(first_module_cells.range1, expected_range1);

    auto second_module_cells = traccc::read_cells_per_module(creader);

    ASSERT_EQ(second_module_cells.items.size(), 8u);

    ASSERT_EQ(second_module_cells.module_id, 1u);
    expected_range0 = std::array<traccc::channel_id,2>{0u,22u};
    ASSERT_EQ(second_module_cells.range0, expected_range0);
    expected_range1 = std::array<traccc::channel_id,2>{4u,98u};
    ASSERT_EQ(second_module_cells.range1, expected_range1);

}

// Google Test can be run manually from the main() function
// or, it can be linked to the gtest_main library for an already
// set-up main() function primed to accept Google Test test cases.
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}