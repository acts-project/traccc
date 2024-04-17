/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <gtest/gtest.h>

#include <functional>
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/utils/cuda/copy.hpp>

#include "tests/cca_test.hpp"
#include "traccc/cuda/clusterization/clusterization_algorithm.hpp"
#include "traccc/cuda/utils/stream.hpp"

namespace {

cca_function_t f = [](const traccc::cell_collection_types::host& cells,
                      const traccc::cell_module_collection_types::host&
                          modules) {
    std::map<traccc::geometry_id, vecmem::vector<traccc::measurement>> result;

    traccc::cuda::stream stream;
    vecmem::cuda::managed_memory_resource mng_mr;
    traccc::memory_resource mr{mng_mr};
    vecmem::cuda::copy copy;
    traccc::cuda::clusterization_algorithm cc(mr, copy, stream, 1024);

    auto measurements_buffer =
        cc(vecmem::get_data(cells), vecmem::get_data(modules));
    traccc::measurement_collection_types::const_device measurements(
        measurements_buffer);

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
    FastSvAlgorithm, ConnectedComponentAnalysisTests,
    ::testing::Combine(
        ::testing::Values(f),
        ::testing::ValuesIn(ConnectedComponentAnalysisTests::get_test_files())),
    ConnectedComponentAnalysisTests::get_test_name);
