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
#include "traccc/alpaka/utils/vecmem_getter.hpp"
#include "traccc/geometry/silicon_detector_description.hpp"

namespace {

// template <TAccTag>
cca_function_t get_f_with(traccc::clustering_config cfg) {
    return [cfg](const traccc::edm::silicon_cell_collection::host& cells,
                 const traccc::silicon_detector_description::host& dd) {
        std::map<traccc::geometry_id, vecmem::vector<traccc::measurement>>
            result;

        auto host_mr = traccc::alpaka::vecmem::get_host_memory_resource<
            traccc::alpaka::AccTag>();
        auto device_mr = traccc::alpaka::vecmem::get_device_memory_resource<
            traccc::alpaka::AccTag>();
        auto copy =
            traccc::alpaka::vecmem::get_device_copy<traccc::alpaka::AccTag>();
        traccc::alpaka::clusterization_algorithm cc({*device_mr}, *copy, cfg);

        traccc::silicon_detector_description::buffer dd_buffer{
            static_cast<
                traccc::silicon_detector_description::buffer::size_type>(
                dd.size()),
            *device_mr};
        copy->setup(dd_buffer)->ignore();
        (*copy)(vecmem::get_data(dd), dd_buffer,
                vecmem::copy::type::host_to_device)
            ->ignore();

        traccc::edm::silicon_cell_collection::buffer cells_buffer{
            static_cast<
                traccc::edm::silicon_cell_collection::buffer::size_type>(
                cells.size()),
            *device_mr};
        copy->setup(cells_buffer)->wait();
        (*copy)(vecmem::get_data(cells), cells_buffer)->wait();

        auto measurements_buffer = cc(cells_buffer, dd_buffer);
        traccc::measurement_collection_types::host measurements{host_mr.get()};
        (*copy)(measurements_buffer, measurements)->wait();

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
