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
#include "traccc/clusterization/clustering_config.hpp"
#include "traccc/clusterization/device/tags.hpp"
#include "traccc/cuda/clusterization/clusterization_algorithm.hpp"
#include "traccc/cuda/utils/stream.hpp"
#include "traccc/geometry/silicon_detector_description.hpp"

namespace {
vecmem::host_memory_resource host_mr;

cca_function_t get_f_with(traccc::clustering_config cfg) {
    return [cfg](const traccc::edm::silicon_cell_collection::host& cells,
                 const traccc::silicon_detector_description::host& dd)
               -> std::pair<std::map<traccc::geometry_id,
                                     traccc::edm::measurement_collection<
                                         traccc::default_algebra>::host>,
                            traccc::edm::silicon_cluster_collection::host> {
        std::map<traccc::geometry_id, traccc::edm::measurement_collection<
                                          traccc::default_algebra>::host>
            geom_to_meas_map;

        traccc::cuda::stream stream;
        vecmem::cuda::device_memory_resource device_mr;
        vecmem::cuda::async_copy copy{stream.cudaStream()};

        traccc::cuda::clusterization_algorithm cc({device_mr, &host_mr}, copy,
                                                  stream, cfg);

        traccc::silicon_detector_description::buffer dd_buffer{
            static_cast<
                traccc::silicon_detector_description::buffer::size_type>(
                dd.size()),
            device_mr};
        copy.setup(dd_buffer)->ignore();
        copy(vecmem::get_data(dd), dd_buffer,
             vecmem::copy::type::host_to_device)
            ->ignore();

        traccc::edm::silicon_cell_collection::buffer cells_buffer{
            static_cast<
                traccc::edm::silicon_cell_collection::buffer::size_type>(
                cells.size()),
            device_mr};
        copy.setup(cells_buffer)->wait();
        copy(vecmem::get_data(cells), cells_buffer)->wait();

        auto [measurements_buffer, cluster_buffer] =
            cc(cells_buffer, dd_buffer,
               traccc::device::clustering_keep_disjoint_set{});
        traccc::edm::measurement_collection<traccc::default_algebra>::host
            measurements{host_mr};
        copy(measurements_buffer, measurements)->wait();

        traccc::edm::silicon_cluster_collection::host clusters{host_mr};
        copy(cluster_buffer, clusters)->wait();

        for (std::size_t i = 0; i < measurements.size(); i++) {
            if (geom_to_meas_map.contains(
                    measurements.at(i).surface_link().value()) == false) {
                geom_to_meas_map.insert(
                    {measurements.at(i).surface_link().value(),
                     traccc::edm::measurement_collection<
                         traccc::default_algebra>::host{host_mr}});
            }
            geom_to_meas_map.at(measurements.at(i).surface_link().value())
                .push_back(measurements.at(i));
        }

        return {std::move(geom_to_meas_map), std::move(clusters)};
    };
}
}  // namespace

TEST_P(ConnectedComponentAnalysisTests, Run) {
    test_connected_component_analysis(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    CUDAFastSvAlgorithm, ConnectedComponentAnalysisTests,
    ::testing::Combine(
        ::testing::Values(get_f_with(default_ccl_test_config())),
        ::testing::ValuesIn(ConnectedComponentAnalysisTests::get_test_files())),
    ConnectedComponentAnalysisTests::get_test_name);

INSTANTIATE_TEST_SUITE_P(
    CUDAFastSvAlgorithmWithScratch, ConnectedComponentAnalysisTests,
    ::testing::Combine(
        ::testing::Values(get_f_with(tiny_ccl_test_config())),
        ::testing::ValuesIn(
            ConnectedComponentAnalysisTests::get_test_files_short())),
    ConnectedComponentAnalysisTests::get_test_name);
