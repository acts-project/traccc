/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Test include(s).
#include "tests/cca_test.hpp"

// Project include(s).
#include "traccc/clusterization/clustering_config.hpp"
#include "traccc/geometry/silicon_detector_description.hpp"
#include "traccc/sycl/clusterization/clusterization_algorithm.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/memory/sycl/device_memory_resource.hpp>
#include <vecmem/utils/sycl/async_copy.hpp>

// Google Test include(s).
#include <gtest/gtest.h>

namespace {

cca_function_t get_f_with(traccc::clustering_config cfg) {
    return [cfg](const traccc::edm::silicon_cell_collection::host& cells,
                 const traccc::silicon_detector_description::host& dd) {
        std::map<traccc::geometry_id, vecmem::vector<traccc::measurement>>
            result;

        vecmem::sycl::queue_wrapper vecmem_queue;
        traccc::sycl::queue_wrapper traccc_queue{vecmem_queue.queue()};
        vecmem::host_memory_resource host_mr;
        vecmem::sycl::device_memory_resource device_mr{vecmem_queue};
        vecmem::sycl::async_copy copy{vecmem_queue};

        traccc::sycl::clusterization_algorithm cc({device_mr, &host_mr}, copy,
                                                  traccc_queue, cfg);

        traccc::silicon_detector_description::buffer dd_buffer{
            static_cast<
                traccc::silicon_detector_description::buffer::size_type>(
                dd.size()),
            device_mr};
        copy.setup(dd_buffer)->ignore();
        copy(vecmem::get_data(dd), dd_buffer,
             vecmem::copy::type::host_to_device)
            ->wait();

        traccc::edm::silicon_cell_collection::buffer cells_buffer{
            static_cast<
                traccc::edm::silicon_cell_collection::buffer::size_type>(
                cells.size()),
            device_mr};
        copy.setup(cells_buffer)->wait();
        copy(vecmem::get_data(cells), cells_buffer)->wait();

        auto measurements_buffer = cc(cells_buffer, dd_buffer);
        traccc::measurement_collection_types::host measurements{&host_mr};
        copy(measurements_buffer, measurements)->wait();

        for (std::size_t i = 0; i < measurements.size(); i++) {
            result[measurements.at(i).surface_link.value()].push_back(
                measurements.at(i));
        }

        return result;
    };
}
}  // namespace

TEST_P(ConnectedComponentAnalysisTests, Run) {
    test_connected_component_analysis(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    SYCLFastSvAlgorithm, ConnectedComponentAnalysisTests,
    ::testing::Combine(
        ::testing::Values(get_f_with(default_ccl_test_config())),
        ::testing::ValuesIn(ConnectedComponentAnalysisTests::get_test_files())),
    ConnectedComponentAnalysisTests::get_test_name);

INSTANTIATE_TEST_SUITE_P(
    SYCLFastSvAlgorithmWithScratch, ConnectedComponentAnalysisTests,
    ::testing::Combine(
        ::testing::Values(get_f_with(tiny_ccl_test_config())),
        ::testing::ValuesIn(
            ConnectedComponentAnalysisTests::get_test_files_short())),
    ConnectedComponentAnalysisTests::get_test_name);
