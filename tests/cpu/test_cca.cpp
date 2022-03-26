/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/clusterization/component_connection.hpp"
#include "traccc/clusterization/measurement_creation.hpp"
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

namespace {
vecmem::host_memory_resource resource;
traccc::component_connection cc(resource);
traccc::measurement_creation mc(resource);
traccc::cell_module module;

std::function<traccc::host_measurement_collection(
    const traccc::host_cell_collection &)>
    fp = traccc::compose(std::function<traccc::host_cluster_container(
                             const traccc::host_cell_collection &)>(
                             std::bind(cc, std::placeholders::_1, module)),
                         std::function<traccc::host_measurement_collection(
                             const traccc::host_cluster_container &)>(
                             std::bind(mc, std::placeholders::_1, module)));

cca_function_t f = [](const traccc::host_cell_container &data) {
    std::map<traccc::geometry_id, std::vector<traccc::measurement>> result;

    for (std::size_t i = 0; i < data.size(); ++i) {
        traccc::host_measurement_collection measurements = fp(data.at(i).items);
        std::vector<traccc::measurement> out(measurements.begin(),
                                             measurements.end());
        result.emplace(data.at(i).header.module, std::move(out));
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
