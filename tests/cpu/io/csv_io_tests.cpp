/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <gtest/gtest.h>

#include <vecmem/memory/host_memory_resource.hpp>

#include "io/csv.hpp"
#include "tests/data_test.hpp"

class io : public traccc::tests::data_test {};

// This defines the local frame test suite
TEST_F(io, csv_read_single_module) {
    std::string file = get_datafile("single_module/cells.csv");
    traccc::cell_reader creader(
        file, {"module", "cannel0", "channel1", "activation", "time"});

    vecmem::host_memory_resource resource;
    auto single_module_cells = traccc::read_cells(creader, resource);
    ASSERT_EQ(single_module_cells.size(), 1u);
    auto module_cells = single_module_cells.at(0).items;
    auto module = single_module_cells.at(0).header;

    ASSERT_EQ(module.module, 0u);
    ASSERT_EQ(module.range0[0], 123u);
    ASSERT_EQ(module.range0[1], 174u);
    ASSERT_EQ(module.range1[0], 32u);
    ASSERT_EQ(module.range1[1], 880u);
    ASSERT_EQ(module_cells.size(), 6u);
}

// This defines the local frame test suite
TEST_F(io, csv_read_two_modules) {
    std::string file = get_datafile("two_modules/cells.csv");

    traccc::cell_reader creader(
        file, {"module", "cannel0", "channel1", "activation", "time"});
    vecmem::host_memory_resource resource;
    auto two_module_cells = traccc::read_cells(creader, resource);
    ASSERT_EQ(two_module_cells.size(), 2u);

    auto first_module_cells = two_module_cells.at(0).items;
    auto first_module = two_module_cells.at(0).header;

    ASSERT_EQ(first_module_cells.size(), 6u);

    ASSERT_EQ(first_module.module, 0u);
    ASSERT_EQ(first_module.range0[0], 123u);
    ASSERT_EQ(first_module.range0[1], 174u);
    ASSERT_EQ(first_module.range1[0], 32u);
    ASSERT_EQ(first_module.range1[1], 880u);

    auto second_module_cells = two_module_cells.at(1).items;
    auto second_module = two_module_cells.at(1).header;

    ASSERT_EQ(second_module_cells.size(), 8u);

    ASSERT_EQ(second_module.module, 1u);
    EXPECT_EQ(second_module.range0[0], 0u);
    EXPECT_EQ(second_module.range0[1], 22u);
    EXPECT_EQ(second_module.range1[0], 4u);
    EXPECT_EQ(second_module.range1[1], 98u);
}

// This reads in the tml pixel barrel first event
TEST_F(io, csv_read_tml_transforms) {
    std::string file = get_datafile("tml_detector/trackml-detector.csv");

    traccc::surface_reader sreader(
        file, {"geometry_id", "cx", "cy", "cz", "rot_xu", "rot_xv", "rot_xw",
               "rot_zu", "rot_zv", "rot_zw"});
    auto tml_barrel_transforms = traccc::read_surfaces(sreader);

    ASSERT_EQ(tml_barrel_transforms.size(), 18751u);
}

// This reads in the tml pixel barrel first event
TEST_F(io, csv_read_tml_pixelbarrel) {
    std::string file =
        get_datafile("tml_pixel_barrel/event000000000-cells.csv");

    traccc::cell_reader creader(
        file, {"module", "cannel0", "channel1", "activation", "time"});
    vecmem::host_memory_resource resource;
    auto tml_barrel_modules = traccc::read_cells(creader, resource);

    ASSERT_EQ(tml_barrel_modules.size(), 2382u);
}

// Google Test can be run manually from the main() function
// or, it can be linked to the gtest_main library for an already
// set-up main() function primed to accept Google Test test cases.
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
