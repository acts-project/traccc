/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/io/mapper.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s).
#include <gtest/gtest.h>

TEST(mappper, particle_map) {
    auto p_map = traccc::generate_particle_map(0, "tml_pixels/ttbar_200/");

    for (auto const& [pid, ptc] : p_map) {
        if (pid == 4503600147464192) {
            EXPECT_EQ(ptc.pos, traccc::vector3({-0.0120002991, -9.48547313e-05,
                                                -15.1165705}));
        }

        else if (pid == 522486832617750528) {
            EXPECT_EQ(ptc.pos,
                      traccc::vector3({-16.0845699, -43.6837349, 33313.7969}));
        }

        else if (pid == 815244996511793152) {
            EXPECT_EQ(ptc.pos, traccc::vector3(
                                   {0.00947579276, 0.00682933675, 59.2630196}));
        }
    }
}

TEST(mappper, hit_particle_map) {

    auto h_p_map = traccc::generate_hit_particle_map(0, "tml_pixels/ttbar_200/",
                                                     "tml_pixels/ttbar_200/");

    for (auto const& [hit, ptc] : h_p_map) {

        if (hit.global == traccc::point3{39.2037048, 0.352969825, -1502.5}) {
            EXPECT_EQ(ptc.particle_id, 58546796263112704);
        }

        else if (hit.global == traccc::point3{65.2966003, -89.9105911, 598}) {
            EXPECT_EQ(ptc.particle_id, 774619137115684864);
        }

        else if (hit.global ==
                 traccc::point3{-631.891235, 3.07215524, 1215.5}) {
            EXPECT_EQ(ptc.particle_id, 369295170719449088);
        }
    }
}

TEST(mappper, hit_map) {

    auto h_map = traccc::generate_hit_map(0, "tml_pixels/ttbar_200/");

    for (auto const& [hid, hit] : h_map) {
        if (hid == 0) {
            EXPECT_EQ(hit.global,
                      traccc::point3({39.2037048, 0.352969825, -1502.5}));
        }

        else if (hid == 74290) {
            EXPECT_EQ(hit.global,
                      traccc::point3({-269.925323, 419.792511, -480.266479}));
        }

        else if (hid == 92349) {
            EXPECT_EQ(hit.global,
                      traccc::point3({-878.646667, -10.9199247, 2952.5}));
        }
    }
}

TEST(mappper, hit_cell_map) {

    auto h_c_map = traccc::generate_hit_cell_map(0, "tml_pixels/ttbar_200/",
                                                 "tml_pixels/ttbar_200/");

    auto compare_cells = [](std::vector<traccc::cell>& cells1,
                            std::vector<traccc::cell>& cells2) {
        std::sort(cells1.begin(), cells1.end());
        std::sort(cells2.begin(), cells2.end());

        EXPECT_EQ(cells1, cells2);
    };

    for (auto [hit, cells] : h_c_map) {
        if (hit.global == traccc::point3{-135.178787, -101.632271, -958}) {
            std::vector<traccc::cell> _cells;
            _cells.push_back({331, 141, 0.025897909, 0});

            compare_cells(cells, _cells);
        }

        else if (hit.global == traccc::point3{-64.083931, -47.5486031, 822.5}) {
            std::vector<traccc::cell> _cells;
            _cells.push_back({277, 992, 0.0143668521, 0});

            compare_cells(cells, _cells);
        }

        else if (hit.global == traccc::point3{823.129333, 517.057373, 1222.5}) {
            std::vector<traccc::cell> _cells;
            _cells.push_back({636, 10, 0.0697815791, 0});
            _cells.push_back({637, 10, 0.186917365, 0});
            _cells.push_back({638, 10, 0.0348203704, 0});

            compare_cells(cells, _cells);
        }
    }
}

TEST(mappper, cell_particle_map) {

    auto c_p_map = traccc::generate_cell_particle_map(
        0, "tml_pixels/ttbar_200/", "tml_pixels/ttbar_200/",
        "tml_pixels/ttbar_200/");

    for (auto const& [c, ptc] : c_p_map) {

        if (c == traccc::cell{222, 1257, 0.0041470062, 0}) {
            EXPECT_EQ(ptc.particle_id, 58546796263112704);
        }

        else if (c == traccc::cell{114, 1403, 0.00306466641, 0}) {
            EXPECT_EQ(ptc.particle_id, 99079195543470080);
        }

        else if (c == traccc::cell{232, 1205, 0.00790336262, 0}) {
            EXPECT_EQ(ptc.particle_id, 702628615719813121);
        }
    }

    vecmem::host_memory_resource mr;
    auto a = traccc::generate_measurement_particle_map(
        0, "/tml_detector/trackml-detector.csv", "tml_pixels/ttbar_200/",
        "tml_pixels/ttbar_200/", "tml_pixels/ttbar_200/", mr);
}
