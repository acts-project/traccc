/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/clusterization/clusterization_algorithm.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/cluster.hpp"
#include "traccc/geometry/detector_description.hpp"

// Test include(s).
#include "tests/cca_test.hpp"

// VecMem include(s).
#include "vecmem/memory/host_memory_resource.hpp"

// GTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <functional>

namespace {
vecmem::host_memory_resource resource;
traccc::host::clusterization_algorithm ca(resource);

cca_function_t f = [](const traccc::cell_collection_types::host& cells,
                      const traccc::detector_description::host& dd) {
    std::map<traccc::geometry_id, vecmem::vector<traccc::measurement>> result;

    const traccc::detector_description::const_data dd_data =
        vecmem::get_data(dd);
    auto measurements = ca(vecmem::get_data(cells), dd_data);
    for (std::size_t i = 0; i < measurements.size(); i++) {
        result[dd.geometry_id().at(measurements.at(i).module_link)].push_back(
            measurements.at(i));
    }

    return result;
};
}  // namespace

TEST_P(ConnectedComponentAnalysisTests, Run) {
    test_connected_component_analysis(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    SparseCclAlgorithm, ConnectedComponentAnalysisTests,
    ::testing::Combine(
        ::testing::Values(f),
        ::testing::ValuesIn(ConnectedComponentAnalysisTests::get_test_files())),
    ConnectedComponentAnalysisTests::get_test_name);
