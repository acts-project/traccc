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

    auto env_d_d = std::getenv("TRACCC_TEST_DATA_DIR");
    if (env_d_d == nullptr)
    {
        throw std::ios_base::failure("Test data directory not found. Please set TRACCC_TEST_DATA_DIR.");
    }
    auto data_directory = std::string(env_d_d) + std::string("/");

    std::string file = data_directory+std::string("single_module/cells.csv");
    traccc::cell_reader creader(file, {"module", "cannel0","channel1","activation","time"} );

    auto single_module_cells = traccc::read_cells(creader);
    ASSERT_EQ(single_module_cells.size(), 1u);
    auto module_cells = single_module_cells[0];

    ASSERT_EQ(module_cells.modcfg.module, 0u);
    auto expected_range0 = std::array<traccc::channel_id,2>{123u,174u};
    ASSERT_EQ(module_cells.modcfg.range0, expected_range0);
    auto expected_range1 = std::array<traccc::channel_id,2>{32u,880u};
    ASSERT_EQ(module_cells.modcfg.range1, expected_range1);
    ASSERT_EQ(module_cells.items.size(), 6u);

}


// This defines the local frame test suite
TEST(io, csv_read_two_modules)
{
    auto env_d_d = std::getenv("TRACCC_TEST_DATA_DIR");
    if (env_d_d == nullptr)
    {
        throw std::ios_base::failure("Test data directory not found. Please set TRACCC_TEST_DATA_DIR.");
    }
    auto data_directory = std::string(env_d_d);
    std::string file = data_directory+std::string("two_modules/cells.csv");
    
    traccc::cell_reader creader(file, {"module", "cannel0","channel1","activation","time"} );
    auto two_module_cells = traccc::read_cells(creader);
    ASSERT_EQ(two_module_cells.size(), 2u);

    auto first_module_cells = two_module_cells[0];
    ASSERT_EQ(first_module_cells.items.size(), 6u);

    ASSERT_EQ(first_module_cells.modcfg.module, 0u);
    auto expected_range0 = std::array<traccc::channel_id,2>{123u,174u};
    ASSERT_EQ(first_module_cells.modcfg.range0, expected_range0);
    auto expected_range1 = std::array<traccc::channel_id,2>{32u,880u};
    ASSERT_EQ(first_module_cells.modcfg.range1, expected_range1);

    auto second_module_cells = two_module_cells[1];

    ASSERT_EQ(second_module_cells.items.size(), 8u);

    ASSERT_EQ(second_module_cells.modcfg.module, 1u);
    expected_range0 = std::array<traccc::channel_id,2>{0u,22u};
    ASSERT_EQ(second_module_cells.modcfg.range0, expected_range0);
    expected_range1 = std::array<traccc::channel_id,2>{4u,98u};
    ASSERT_EQ(second_module_cells.modcfg.range1, expected_range1);

}


// This reads in the tml pixel barrel first event
TEST(io, csv_read_tml_transforms){

    auto env_d_d = std::getenv("TRACCC_TEST_DATA_DIR");
    if (env_d_d == nullptr)
    {
        throw std::ios_base::failure("Test data directory not found. Please set TRACCC_TEST_DATA_DIR.");
    }
    auto data_directory = std::string(env_d_d);
    std::string file = data_directory+std::string("tml_detector/trackml-detector.csv");

    traccc::surface_reader sreader(file, {"geometry_id", "cx", "cy", "cz", "rot_xu", "rot_xv", "rot_xw", "rot_zu", "rot_zv", "rot_zw"} );
    auto tml_barrel_transforms = traccc::read_surfaces(sreader); 
    
    ASSERT_EQ(tml_barrel_transforms.size(), 18751u) ;

}

// This reads in the tml pixel barrel first event
TEST(io, csv_read_tml_pixelbarrel){

    auto env_d_d = std::getenv("TRACCC_TEST_DATA_DIR");
    if (env_d_d == nullptr)
    {
        throw std::ios_base::failure("Test data directory not found. Please set TRACCC_TEST_DATA_DIR.");
    }
    auto data_directory = std::string(env_d_d);
    std::string file = data_directory+std::string("tml_pixel_barrel/event000000000-cells.csv");

    traccc::cell_reader creader(file, {"module", "cannel0","channel1","activation","time"} );
    auto tml_barrel_modules = traccc::read_cells(creader); 
    
    ASSERT_EQ(tml_barrel_modules.size(), 2382u) ;

}

// Google Test can be run manually from the main() function
// or, it can be linked to the gtest_main library for an already
// set-up main() function primed to accept Google Test test cases.
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
