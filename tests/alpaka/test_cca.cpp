/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <gtest/gtest.h>

#include <functional>
#include <vecmem/memory/host_memory_resource.hpp>

#include "tests/cca_test.hpp"
#include "traccc/alpaka/clusterization/clusterization_algorithm.hpp"

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#include <vecmem/utils/cuda/copy.hpp>
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#include <vecmem/memory/hip/device_memory_resource.hpp>
#include <vecmem/memory/hip/host_memory_resource.hpp>
#include <vecmem/utils/hip/copy.hpp>
#endif

namespace {

cca_function_t get_f_with(traccc::clustering_config cfg) {
    return [cfg](const traccc::cell_collection_types::host& cells,
                 const traccc::cell_module_collection_types::host& modules) {
        std::map<traccc::geometry_id, vecmem::vector<traccc::measurement>>
            result;

        vecmem::host_memory_resource host_mr;

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
        vecmem::cuda::copy copy;
        vecmem::cuda::device_memory_resource device_mr;
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
        vecmem::hip::copy copy;
        vecmem::hip::device_memory_resource device_mr;
#else
        vecmem::copy copy;
        vecmem::host_memory_resource device_mr;
#endif

        traccc::alpaka::clusterization_algorithm cc({device_mr}, copy, cfg);

        traccc::cell_collection_types::buffer cells_buffer{
            static_cast<traccc::cell_collection_types::buffer::size_type>(
                cells.size()),
            device_mr};
        copy.setup(cells_buffer)->wait();
        copy(vecmem::get_data(cells), cells_buffer)->wait();

        traccc::cell_module_collection_types::buffer modules_buffer{
            static_cast<
                traccc::cell_module_collection_types::buffer::size_type>(
                modules.size()),
            device_mr};
        copy.setup(modules_buffer)->wait();
        copy(vecmem::get_data(modules), modules_buffer)->wait();

        auto measurements_buffer = cc(cells_buffer, modules_buffer);
        traccc::measurement_collection_types::host measurements{&host_mr};
        copy(measurements_buffer, measurements)->wait();

        for (std::size_t i = 0; i < measurements.size(); i++) {
            result[modules.at(measurements.at(i).module_link)
                       .surface_link.value()]
                .push_back(measurements.at(i));
        }

        return result;
    };
}
}  // namespace

TEST_P(ConnectedComponentAnalysisTests, Run) {
    test_connected_component_analysis(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    AlpakaFastSvAlgorithm, ConnectedComponentAnalysisTests,
    ::testing::Combine(
        ::testing::Values(get_f_with(default_ccl_test_config())),
        ::testing::ValuesIn(ConnectedComponentAnalysisTests::get_test_files())),
    ConnectedComponentAnalysisTests::get_test_name);

INSTANTIATE_TEST_SUITE_P(
    AlpakaFastSvAlgorithmWithScratch, ConnectedComponentAnalysisTests,
    ::testing::Combine(
        ::testing::Values(get_f_with(tiny_ccl_test_config())),
        ::testing::ValuesIn(
            ConnectedComponentAnalysisTests::get_test_files_short())),
    ConnectedComponentAnalysisTests::get_test_name);
