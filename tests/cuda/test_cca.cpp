/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <gtest/gtest.h>

#include <functional>

#include "tests/cca_test.hpp"
#include "traccc/cuda/cca/component_connection.hpp"

namespace {
traccc::cuda::component_connection cc;

cca_function_t f = [](const traccc::host_cell_container &data) {
    std::map<traccc::geometry_id, std::vector<traccc::measurement>> result;

    traccc::host_measurement_container mss = cc(data);

    for (std::size_t i = 0; i < mss.size(); ++i) {
        std::vector<traccc::measurement> msv;

        for (std::size_t j = 0; j < mss.at(i).items.size(); ++j) {
            msv.push_back(mss.at(i).items.at(j));
        }

        result.emplace(mss.at(i).header.module, std::move(msv));
    }

    return result;
};
}  // namespace

TEST_P(ConnectedComponentAnalysisTests, Run) {
    test_connected_component_analysis(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    FastSvAlgorithm, ConnectedComponentAnalysisTests,
    ::testing::Combine(
        ::testing::Values(f),
        ::testing::ValuesIn(ConnectedComponentAnalysisTests::get_test_files())),
    ConnectedComponentAnalysisTests::get_test_name);
