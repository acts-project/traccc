/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <gtest/gtest.h>

#include <functional>
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>

#include "tests/cca_test.hpp"
#include "traccc/cuda/clusterization/clusterization_algorithm.hpp"
#include "traccc/cuda/utils/stream.hpp"

namespace {

cca_function_t f = [](const traccc::cell_collection_types::host& cells,
                      const traccc::cell_module_collection_types::host&
                          modules) {
    std::map<traccc::geometry_id, vecmem::vector<traccc::measurement>> result;

    vecmem::host_memory_resource host_mr;
    traccc::cuda::stream stream;
    vecmem::cuda::device_memory_resource device_mr;
    vecmem::cuda::async_copy copy{stream.cudaStream()};

    traccc::cuda::clusterization_algorithm cc({device_mr}, copy, stream, 1024);

    traccc::cell_collection_types::buffer cells_buffer{
        static_cast<traccc::cell_collection_types::buffer::size_type>(
            cells.size()),
        device_mr};
    copy.setup(cells_buffer);
    copy(vecmem::get_data(cells), cells_buffer)->ignore();

    traccc::cell_module_collection_types::buffer modules_buffer{
        static_cast<traccc::cell_module_collection_types::buffer::size_type>(
            modules.size()),
        device_mr};
    copy.setup(modules_buffer);
    copy(vecmem::get_data(modules), modules_buffer)->ignore();

    auto measurements_buffer = cc(cells_buffer, modules_buffer);
    traccc::measurement_collection_types::host measurements{&host_mr};
    copy(measurements_buffer, measurements)->wait();

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
