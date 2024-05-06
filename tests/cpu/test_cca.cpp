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

// Test include(s).
#include "tests/cca_test.hpp"

// VecMem include(s).
#include "vecmem/memory/host_memory_resource.hpp"

// GTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <functional>

/*
@FIXME: Change the csv data format for csv
namespace {
vecmem::host_memory_resource resource;
traccc::host::clusterization_algorithm ca(resource);

cca_function_t f = [](const traccc::cell_collection_types::host& cells,
                      const traccc::cell_module_collection_types::host&
                          modules) {
    std::map<traccc::geometry_id, vecmem::vector<traccc::measurement>> result;

    auto measurements = ca(vecmem::get_data(cells), vecmem::get_data(modules));
    for (std::size_t i = 0; i < measurements.size(); i++) {
        result[modules.at(measurements.at(i).module_link).surface_link.value()]
            .push_back(measurements.at(i));
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
*/