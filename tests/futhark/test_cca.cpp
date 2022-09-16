/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/futhark/component_connection.hpp"

// Test include(s).
#include "tests/cca_test.hpp"

// GTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <functional>

namespace {
vecmem::host_memory_resource resource;
traccc::futhark::component_connection cc(resource);

cca_function_t f = [](const traccc::cell_container_types::host &data) {
    std::map<traccc::geometry_id, vecmem::vector<traccc::measurement>> result;

    traccc::measurement_container_types::host mss = cc(data);

    for (std::size_t i = 0; i < mss.size(); ++i) {
        vecmem::vector<traccc::measurement> msv(&resource);

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
    FutharkCcaAlgorithm, ConnectedComponentAnalysisTests,
    ::testing::Combine(
        ::testing::Values(f),
        ::testing::ValuesIn(ConnectedComponentAnalysisTests::get_test_files())),
    ConnectedComponentAnalysisTests::get_test_name);
