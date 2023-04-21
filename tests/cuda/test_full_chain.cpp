/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/clusterization/clusterization_algorithm.hpp"
#include "traccc/clusterization/spacepoint_formation.hpp"
#include "traccc/cuda/clusterization/clusterization_algorithm.hpp"
#include "traccc/cuda/seeding/seeding_algorithm.hpp"
#include "traccc/cuda/seeding/track_params_estimation.hpp"
#include "traccc/cuda/utils/stream.hpp"
#include "traccc/io/read.hpp"
#include "traccc/performance/collection_comparator.hpp"
#include "traccc/seeding/seeding_algorithm.hpp"
#include "traccc/seeding/track_params_estimation.hpp"

// VecMem include(s).
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>

// GTest include(s).
#include <gtest/gtest.h>

using namespace traccc;

class FullChainTests
    : public ::testing::TestWithParam<
          std::tuple<std::string, std::string, std::string, unsigned int>> {};

// This defines the local frame test suite
TEST_P(FullChainTests, Run) {

    static constexpr scalar UNCERTAINTY = 0.01;
    static constexpr scalar ACCURACY = 0.99;

    const std::string detector_file = std::get<0>(GetParam());
    const std::string digitization_config_file = std::get<1>(GetParam());
    const std::string input_directory = std::get<2>(GetParam());
    const unsigned int n_events = std::get<3>(GetParam());

    // Memory resources used by the application.
    vecmem::host_memory_resource host_mr;
    vecmem::cuda::host_memory_resource cuda_host_mr;
    vecmem::cuda::device_memory_resource device_mr;
    traccc::memory_resource mr{device_mr, &cuda_host_mr};

    // CPU algorithms
    clusterization_algorithm ca(host_mr);
    spacepoint_formation sf(host_mr);
    seeding_algorithm sa(host_mr);
    track_params_estimation tp(host_mr);

    traccc::cuda::stream stream;
    vecmem::cuda::async_copy copy{stream.cudaStream()};

    // GPU algorithms
    traccc::cuda::clusterization_algorithm ca_cuda(mr, copy, stream, 1024u);
    traccc::cuda::seeding_algorithm sa_cuda(mr, copy, stream);
    traccc::cuda::track_params_estimation tp_cuda(mr, copy, stream);

    // Read input data
    demonstrator_input inputVec(&host_mr);
    // Create empty inputs using the correct memory resource
    for (std::size_t i = 0; i < n_events; ++i) {
        inputVec.push_back(demonstrator_input::value_type(&host_mr));
    }
    io::read(inputVec, n_events, input_directory, detector_file,
             digitization_config_file, data_format::csv);

    for (unsigned int event = 0; event < n_events; ++event) {
        auto& cells = inputVec[event].cells;
        auto& modules = inputVec[event].modules;

        // Run CPU chain
        auto spacepoints = sf(ca(cells, modules), modules);
        auto seeds = sa(spacepoints);
        auto track_params = tp(spacepoints, seeds);

        // Copy input to device
        cell_collection_types::buffer cells_buffer(cells.size(), mr.main);
        copy(vecmem::get_data(cells), cells_buffer);
        cell_module_collection_types::buffer modules_buffer(modules.size(),
                                                            mr.main);
        copy(vecmem::get_data(modules), modules_buffer);

        // Run GPU chain
        auto spacepoints_buffer = ca_cuda(cells_buffer, modules_buffer).first;
        auto seeds_buffer = sa_cuda(spacepoints_buffer);
        auto track_params_buffer = tp_cuda(spacepoints_buffer, seeds_buffer);

        // Copy output to host
        spacepoint_collection_types::host spacepoints_cuda;
        seed_collection_types::host seeds_cuda;
        bound_track_parameters_collection_types::host track_params_cuda;

        copy(spacepoints_buffer, spacepoints_cuda);
        copy(seeds_buffer, seeds_cuda);
        copy(track_params_buffer, track_params_cuda);

        // Wait for everything
        stream.synchronize();

        // Compare the spacepoints made on the host and on the device.
        traccc::collection_comparator<traccc::spacepoint> compare_spacepoints{
            "spacepoints"};
        auto cp_sp = compare_spacepoints.compare(
            vecmem::get_data(spacepoints), vecmem::get_data(spacepoints_cuda),
            UNCERTAINTY);

        // Compare the seeds made on the host and on the device
        traccc::collection_comparator<traccc::seed> compare_seeds{
            "seeds", traccc::details::comparator_factory<traccc::seed>{
                         vecmem::get_data(spacepoints),
                         vecmem::get_data(spacepoints_cuda)}};
        auto cp_sd = compare_seeds.compare(
            vecmem::get_data(seeds), vecmem::get_data(seeds_cuda), UNCERTAINTY);

        // Compare the track parameters made on the host and on the device.
        traccc::collection_comparator<traccc::bound_track_parameters>
            compare_track_parameters{"track parameters"};
        auto cp_tp = compare_track_parameters.compare(
            vecmem::get_data(track_params), vecmem::get_data(track_params_cuda),
            UNCERTAINTY);

        // Check if obtained accuracy above minimum threshold.
        EXPECT_GE(cp_sp, ACCURACY);
        EXPECT_GE(cp_sd, ACCURACY);
        EXPECT_GE(cp_tp, ACCURACY);
    }
}

INSTANTIATE_TEST_SUITE_P(
    FullChainValidation, FullChainTests,
    ::testing::Values(
        std::make_tuple("tml_detector/trackml-detector.csv",
                        "tml_detector/default-geometric-config-generic.json",
                        "tml_full/ttbar_mu20/", 1),
        std::make_tuple("tml_detector/trackml-detector.csv",
                        "tml_detector/default-geometric-config-generic.json",
                        "tml_full/ttbar_mu200/", 1)));
