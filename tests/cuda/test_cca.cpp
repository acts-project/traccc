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

cca_function_t f =
    [](const traccc::cell_collection_types::host& cells,
       const traccc::cell_module_collection_types::host& modules) {
        std::map<traccc::geometry_id, vecmem::vector<traccc::alt_measurement>>
            result;

        auto measurements = cc(cells);

        for (std::size_t i = 0; i < measurements.size(); i++) {
            result[modules.at(measurements.at(i).module_link).module].push_back(
                measurements.at(i));
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