/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/clusterization/clustering_config.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/silicon_cell_collection.hpp"
#include "traccc/geometry/silicon_detector_description.hpp"
#include "traccc/io/read_cells.hpp"

// Test include(s).
#include "tests/data_test.hpp"

// VecMem include(s).
#include <vecmem/containers/vector.hpp>
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s).
#include <gtest/gtest.h>

// DFE include(s).
#include <dfe/dfe_io_dsv.hpp>
#include <dfe/dfe_namedtuple.hpp>

// System include(s).
#include <functional>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>

using cca_function_t = std::function<
    std::map<traccc::geometry_id, vecmem::vector<traccc::measurement>>(
        const traccc::edm::silicon_cell_collection::host &,
        const traccc::silicon_detector_description::host &)>;

inline traccc::clustering_config default_ccl_test_config() {
    traccc::clustering_config rv;

    rv.threads_per_partition = 128;
    rv.max_cells_per_thread = 16;
    rv.target_cells_per_thread = 8;
    rv.backup_size_multiplier = 256;

    return rv;
}

inline traccc::clustering_config tiny_ccl_test_config() {
    traccc::clustering_config rv;

    rv.threads_per_partition = 128;
    rv.max_cells_per_thread = 1;
    rv.target_cells_per_thread = 1;
    rv.backup_size_multiplier = 16384;

    return rv;
}

class ConnectedComponentAnalysisTests
    : public traccc::tests::data_test,
      public testing::WithParamInterface<
          std::tuple<cca_function_t, std::string>> {
    public:
    struct cca_truth_hit {
        uint64_t geometry_id = 0;
        uint64_t measurement_id = 0;
        uint64_t num_cells = 0;
        traccc::scalar channel0 = 0;
        traccc::scalar channel1 = 0;
        traccc::scalar variance0 = 0.f;
        traccc::scalar variance1 = 0.f;

        DFE_NAMEDTUPLE(cca_truth_hit, geometry_id, measurement_id, num_cells,
                       channel0, channel1, variance0, variance1);
    };

    using cca_truth_hit_reader = dfe::NamedTupleCsvReader<cca_truth_hit>;

    inline static std::string get_test_name(
        const testing::TestParamInfo<ParamType> &info) {
        return std::get<1>(info.param);
    }

    inline static std::vector<std::string> get_test_files(void) {
        const std::vector<std::pair<std::string, std::size_t>> cases = {
            {"dense", 100},
            {"multiple_module_single_hit", 100},
            {"single_module_multiple_hit_single", 100},
            {"single_module_multiple_hit_single_sparse", 100},
            {"single_module_single_hit", 100},
            {"very_dense", 100},
            {"trackml_like", 30},
        };
        std::vector<std::string> out;

        for (const std::pair<std::string, std::size_t> &c : cases) {
            for (std::size_t i = 0; i < c.second; ++i) {
                std::ostringstream ss;
                ss << c.first << "_" << std::setfill('0') << std::setw(10) << i;
                out.push_back(ss.str());
            }
        }

        return out;
    }

    inline static std::vector<std::string> get_test_files_short(void) {
        const std::vector<std::pair<std::string, std::size_t>> cases = {
            {"trackml_like", 10},
        };
        std::vector<std::string> out;

        for (const std::pair<std::string, std::size_t> &c : cases) {
            for (std::size_t i = 0; i < c.second; ++i) {
                std::ostringstream ss;
                ss << c.first << "_" << std::setfill('0') << std::setw(10) << i;
                out.push_back(ss.str());
            }
        }

        return out;
    }

    inline void test_connected_component_analysis(ParamType p) {
        cca_function_t f = std::get<0>(p);
        std::string file_prefix = std::get<1>(p);

        std::string file_hits =
            get_datafile("cca_test/" + file_prefix + "_hits.csv");
        std::string file_truth =
            get_datafile("cca_test/" + file_prefix + "_truth.csv");

        // Host memory resource for the test.
        vecmem::host_memory_resource mr;

        // Create a dummy detector description. With a description of enough
        // detector modules for all the input files that the test uses.
        static constexpr std::size_t NMODULES = 2500;
        static constexpr traccc::scalar pitch = 1.f;
        traccc::silicon_detector_description::host dd{mr};
        dd.resize(NMODULES);
        for (std::size_t i = 0; i < NMODULES; ++i) {
            dd.geometry_id()[i] = detray::geometry::barcode{i};
            dd.acts_geometry_id()[i] = i;
            dd.reference_x()[i] = -0.5f;
            dd.reference_y()[i] = -0.5f;
            dd.pitch_x()[i] = pitch;
            dd.pitch_y()[i] = pitch;
        }

        traccc::edm::silicon_cell_collection::host cells{mr};
        traccc::io::read_cells(cells, file_hits,
                               traccc::getDummyLogger().clone(), &dd);

        std::map<traccc::geometry_id, vecmem::vector<traccc::measurement>>
            result = f(cells, dd);

        std::size_t total_truth = 0, total_found = 0;

        for (const auto &i : result) {
            total_found += i.second.size();
        }

        cca_truth_hit_reader truth_reader(file_truth);

        traccc::scalar var_adjustment = (pitch * pitch) / 12.f;

        cca_truth_hit io_truth;
        while (truth_reader.read(io_truth)) {
            ASSERT_TRUE(result.find(io_truth.geometry_id) != result.end());

            const vecmem::vector<traccc::measurement> &meas =
                result.at(io_truth.geometry_id);

            const traccc::scalar tol = 0.0001f;

            auto match = std::find_if(
                meas.begin(), meas.end(),
                [&io_truth, tol](const traccc::measurement &i) {
                    return std::abs(i.local[0] - io_truth.channel0) < tol &&
                           std::abs(i.local[1] - io_truth.channel1) < tol;
                });

            ASSERT_TRUE(match != meas.end());

            EXPECT_NEAR(match->local[0], io_truth.channel0, tol);
            EXPECT_NEAR(match->local[1], io_truth.channel1, tol);
            EXPECT_NEAR(match->variance[0], io_truth.variance0 + var_adjustment,
                        tol);
            EXPECT_NEAR(match->variance[1], io_truth.variance1 + var_adjustment,
                        tol);

            ++total_truth;
        }

        EXPECT_EQ(total_truth, total_found);
    }
};
