/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <gtest/gtest.h>

#include <functional>

#include "clusterization/component_connection.hpp"
#include "clusterization/measurement_creation.hpp"
#include "definitions/primitives.hpp"
#include "edm/cell.hpp"
#include "edm/cluster.hpp"
#include "tests/cca_test.hpp"
#include "vecmem/memory/host_memory_resource.hpp"

namespace {
vecmem::host_memory_resource resource;
traccc::component_connection cc(resource);
traccc::measurement_creation mc(resource);
traccc::cell_module module;

std::function<traccc::host_measurement_collection(
    const traccc::host_cell_collection &)>
    fp = traccc::compose(std::function<traccc::cluster_collection(
                             const traccc::host_cell_collection &)>(
                             std::bind(cc, std::placeholders::_1, module)),
                         std::function<traccc::host_measurement_collection(
                             const traccc::cluster_collection &)>(
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
