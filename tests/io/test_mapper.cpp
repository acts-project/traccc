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

std::size_t event = 0;

std::string detector_file = "/tml_detector/trackml-detector.csv";
std::string hits_dir = "tml_pixels/ttbar_200/";
std::string cells_dir = "tml_pixels/ttbar_200/";
std::string particles_dir = "tml_pixels/ttbar_200/";

std::string mock_hits_dir = "../tests/io/mock_data/";
std::string mock_cells_dir = "../tests/io/mock_data/";
std::string mock_particles_dir = "../tests/io/mock_data/";

/***
 * Simulation data test
 */

// Test generate_particle_map function
TEST(mappper, particle_map) {
    auto p_map = traccc::generate_particle_map(event, particles_dir);

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

// Test generate_hit_particle_map function
TEST(mappper, hit_particle_map) {
    auto h_p_map =
        traccc::generate_hit_particle_map(event, hits_dir, particles_dir);

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

// Test generate_hit_map function
TEST(mappper, hit_map) {
    auto h_map = traccc::generate_hit_map(event, hits_dir);

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

// Test generate_hit_cell_map function
TEST(mappper, hit_cell_map) {
    auto h_c_map = traccc::generate_hit_cell_map(event, cells_dir, hits_dir);

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

// Test generate_cell_particle_map function
TEST(mappper, cell_particle_map) {
    auto c_p_map = traccc::generate_cell_particle_map(event, cells_dir,
                                                      hits_dir, particles_dir);

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
}

/***
 * Mock data test
 *
 * Mock data consists of three particles each of which has one hit
 *
 * first particle: one hit, three cells
 * second particle: one hit, four cells
 * thrid particle: one hit, three cells
 *
 * [ ] [1] [ ] [ ] [ ] [ ] [ ] [ ]
 * [1][1,2][2] [ ] [ ] [ ] [ ] [ ]
 * [ ] [2] [2] [ ] [ ] [ ] [ ] [ ]
 * [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ]
 * [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ]
 * [ ] [ ] [ ] [ ] [ ] [3] [3] [3]
 * [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ]
 * [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ]
 *
 * Current traccc's CCA algorithm will make two clusters
 *
 * first cluster is a set of cells generated by 1st and 2nd particles
 * second cluster is a set of cells generated by 3rd particle
 *
 */

// Test generate_particle_map function
TEST(mappper, mock_particle_map) {
    auto p_map = traccc::generate_particle_map(event, mock_particles_dir);

    EXPECT_EQ(p_map.size(), 3);

    std::size_t pid0 = 4503599644147712;
    std::size_t pid1 = 4503599660924928;
    std::size_t pid2 = 4503599744811008;

    EXPECT_EQ(p_map[pid0].pos,
              traccc::vector3({-0.0120002991, -9.48547313e-05, -15.1165705}));

    EXPECT_EQ(p_map[pid1].pos, traccc::vector3({-0.4, -0.23412, -100.}));

    EXPECT_EQ(p_map[pid2].pos, traccc::vector3({-1999., -9.231, 23425.}));
}

// Test generate_hit_particle_map function
TEST(mappper, mock_hit_particle_map) {
    auto h_p_map = traccc::generate_hit_particle_map(event, mock_hits_dir,
                                                     mock_particles_dir);

    EXPECT_EQ(h_p_map.size(), 3);

    traccc::spacepoint sp0;
    sp0.global = traccc::point3{39.2037048, 0.352969825, -1502.5};

    traccc::spacepoint sp1;
    sp1.global = traccc::point3{31.5048428, 5.16000509, -1502.5};

    traccc::spacepoint sp2;
    sp2.global = traccc::point3{90.4015808, 0.7420941, -1502.5};

    EXPECT_EQ(h_p_map[sp0].particle_id, 4503599644147712);
    EXPECT_EQ(h_p_map[sp1].particle_id, 4503599660924928);
    EXPECT_EQ(h_p_map[sp2].particle_id, 4503599744811008);
}

// Test generate_hit_map function
TEST(mappper, mock_hit_map) {
    auto h_map = traccc::generate_hit_map(event, mock_hits_dir);

    EXPECT_EQ(h_map.size(), 3);

    h_map[0].global = traccc::point3{39.2037048, 0.352969825, -1502.5};
    h_map[1].global = traccc::point3{-269.925323, 419.792511, -480.266479};
    h_map[2].global = traccc::point3{-878.646667, -10.9199247, 2952.5};
}

// Test generate_hit_cell_map function
TEST(mappper, mock_hit_cell_map) {
    auto h_c_map =
        traccc::generate_hit_cell_map(event, mock_cells_dir, mock_hits_dir);

    EXPECT_EQ(h_c_map.size(), 3);

    traccc::spacepoint sp0;
    sp0.global = traccc::point3{39.2037048, 0.352969825, -1502.5};

    traccc::spacepoint sp1;
    sp1.global = traccc::point3{31.5048428, 5.16000509, -1502.5};

    traccc::spacepoint sp2;
    sp2.global = traccc::point3{90.4015808, 0.7420941, -1502.5};

    std::vector<traccc::cell> cells0;
    cells0.push_back({1, 0, 0.0041470062, 0});
    cells0.push_back({0, 1, 0.00306466641, 0});
    cells0.push_back({1, 1, 0.00868905429, 0});

    std::vector<traccc::cell> cells1;
    cells1.push_back({1, 1, 0.00886478275, 0});
    cells1.push_back({1, 2, 0.00580428448, 0});
    cells1.push_back({2, 1, 0.0016894876, 0});
    cells1.push_back({2, 2, 0.00199076766, 0});

    std::vector<traccc::cell> cells2;
    cells2.push_back({5, 5, 0.00632160669, 0});
    cells2.push_back({5, 6, 0.00911649223, 0});
    cells2.push_back({5, 7, 0.00518329488, 0});

    EXPECT_EQ(h_c_map[sp0], cells0);
    EXPECT_EQ(h_c_map[sp1], cells1);
    EXPECT_EQ(h_c_map[sp2], cells2);
}

// Test generate_cell_particle_map function
TEST(mappper, mock_cell_particle_map) {

    auto c_p_map = traccc::generate_cell_particle_map(
        event, mock_cells_dir, mock_hits_dir, mock_particles_dir);

    std::vector<traccc::cell> cells;
    traccc::cell cell0{1, 0, 0.0041470062, 0};
    traccc::cell cell1{0, 1, 0.00306466641, 0};
    traccc::cell cell2{1, 1, 0.00868905429, 0};
    traccc::cell cell3{1, 1, 0.00886478275, 0};
    traccc::cell cell4{1, 2, 0.00580428448, 0};
    traccc::cell cell5{2, 1, 0.0016894876, 0};
    traccc::cell cell6{2, 2, 0.00199076766, 0};
    traccc::cell cell7{5, 5, 0.00632160669, 0};
    traccc::cell cell8{5, 6, 0.00911649223, 0};
    traccc::cell cell9{5, 7, 0.00518329488, 0};

    EXPECT_EQ(c_p_map[cell0].particle_id, 4503599644147712);
    EXPECT_EQ(c_p_map[cell1].particle_id, 4503599644147712);
    EXPECT_EQ(c_p_map[cell2].particle_id, 4503599644147712);
    EXPECT_EQ(c_p_map[cell3].particle_id, 4503599660924928);
    EXPECT_EQ(c_p_map[cell4].particle_id, 4503599660924928);
    EXPECT_EQ(c_p_map[cell5].particle_id, 4503599660924928);
    EXPECT_EQ(c_p_map[cell6].particle_id, 4503599660924928);
    EXPECT_EQ(c_p_map[cell7].particle_id, 4503599744811008);
    EXPECT_EQ(c_p_map[cell8].particle_id, 4503599744811008);
    EXPECT_EQ(c_p_map[cell9].particle_id, 4503599744811008);
}

// Test generate_measurement_cell_map function
TEST(mappper, mock_measurement_cell_map) {

    auto compare_cells = [](std::vector<traccc::cell>& cells1,
                            std::vector<traccc::cell>& cells2) {
        std::sort(cells1.begin(), cells1.end());
        std::sort(cells2.begin(), cells2.end());

        EXPECT_EQ(cells1, cells2);
    };

    vecmem::host_memory_resource resource;

    auto m_c_map = traccc::generate_measurement_cell_map(
        event, detector_file, mock_cells_dir, resource);

    std::vector<traccc::cell> cells0;
    cells0.push_back({1, 0, 0.0041470062, 0});
    cells0.push_back({0, 1, 0.00306466641, 0});
    cells0.push_back({1, 1, 0.00868905429, 0});
    cells0.push_back({1, 1, 0.00886478275, 0});
    cells0.push_back({1, 2, 0.00580428448, 0});
    cells0.push_back({2, 1, 0.0016894876, 0});
    cells0.push_back({2, 2, 0.00199076766, 0});

    std::vector<traccc::cell> cells1;
    cells1.push_back({5, 5, 0.00632160669, 0});
    cells1.push_back({5, 6, 0.00911649223, 0});
    cells1.push_back({5, 7, 0.00518329488, 0});

    EXPECT_EQ(m_c_map.size(), 2);

    bool has_first_cluster = false;
    bool has_second_cluster = false;

    for (auto [meas, cells] : m_c_map) {

        // first cluster with seven cells
        if (cells.size() == 7) {
            compare_cells(cells, cells0);
            has_first_cluster = true;
        }

        // second cluster with three cells
        else if (cells.size() == 3) {
            compare_cells(cells, cells1);
            has_second_cluster = true;
        }
    }

    EXPECT_EQ(has_first_cluster, true);
    EXPECT_EQ(has_second_cluster, true);
}

// Test the first generate_measurement_particle_map function
TEST(mappper, mock_measurement_particle_map_with_clusterization) {

    vecmem::host_memory_resource mr;

    auto m_p_map = traccc::generate_measurement_particle_map(
        event, detector_file, mock_cells_dir, mock_hits_dir, mock_particles_dir,
        mr);

    // There are two measurements (or clusters)
    EXPECT_EQ(m_p_map.size(), 2);

    bool has_first_cluster = false;
    bool has_second_cluster = false;

    bool has_first_particle = false;
    bool has_second_particle = false;
    bool has_third_particle = false;

    for (auto const& [meas, ptcs] : m_p_map) {

        // first measurement (or cluster) is contributed by 1st and 2nd
        // particles
        if (ptcs.size() == 2) {
            for (auto const& [ptc, count] : ptcs) {
                if (ptc.particle_id == 4503599644147712) {
                    // number of cells from 1st particle
                    EXPECT_EQ(count, 3);
                    has_first_particle = true;
                } else if (ptc.particle_id == 4503599660924928) {
                    // number of cells from 2nd particle
                    EXPECT_EQ(count, 4);
                    has_second_particle = true;
                }
            }

            has_first_cluster = true;
        }

        // second measurement (or cluster) is contributed by 3rd particle
        else if (ptcs.size() == 1) {
            for (auto const& [ptc, count] : ptcs) {
                if (ptc.particle_id == 4503599744811008) {
                    // number of cells from 3rd particle
                    EXPECT_EQ(count, 3);
                    has_third_particle = true;
                }
            }

            has_second_cluster = true;
        }
    }

    EXPECT_EQ(has_first_cluster, true);
    EXPECT_EQ(has_second_cluster, true);
    EXPECT_EQ(has_first_particle, true);
    EXPECT_EQ(has_second_particle, true);
    EXPECT_EQ(has_third_particle, true);
}

// Test the second generate_measurement_particle_map function
TEST(mappper, mock_measurement_particle_map_without_clusterization) {

    vecmem::host_memory_resource mr;

    auto m_p_map = traccc::generate_measurement_particle_map(
        event, detector_file, mock_hits_dir, mock_particles_dir, mr);

    // Without clusterization, there are three measurements each of which is
    // from each particle
    EXPECT_EQ(m_p_map.size(), 3);

    bool has_first_particle = false;
    bool has_second_particle = false;
    bool has_third_particle = false;

    for (auto const& [meas, ptcs] : m_p_map) {
        // Without clusterization, there is only one contributing particle for
        // measurement
        EXPECT_EQ(ptcs.size(), 1);

        for (auto const& [ptc, count] : ptcs) {

            if (ptc.particle_id == 4503599644147712) {
                EXPECT_EQ(count, 1);
                has_first_particle = true;
            } else if (ptc.particle_id == 4503599660924928) {
                EXPECT_EQ(count, 1);
                has_second_particle = true;
            } else if (ptc.particle_id == 4503599744811008) {
                EXPECT_EQ(count, 1);
                has_third_particle = true;
            }
        }
    }

    EXPECT_EQ(has_first_particle, true);
    EXPECT_EQ(has_second_particle, true);
    EXPECT_EQ(has_third_particle, true);
}
