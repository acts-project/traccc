/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "detray/core/detector.hpp"
#include "detray/definitions/units.hpp"

// Detray IO include(s)
#include "detray/io/frontend/detector_reader.hpp"

// Detray test include(s)
#include "detray/options/detector_io_options.hpp"
#include "detray/options/parse_options.hpp"
#include "detray/options/propagation_options.hpp"
#include "detray/options/track_generator_options.hpp"
#include "detray/test/common/detail/register_checks.hpp"
#include "detray/test/cpu/material_scan.hpp"
#include "detray/test/device/cuda/material_validation.hpp"

// Vecmem include(s)
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s)
#include <gtest/gtest.h>

// Boost
#include "detray/options/boost_program_options.hpp"

// System include(s)
#include <sstream>
#include <stdexcept>
#include <string>

namespace po = boost::program_options;

using namespace detray;

int main(int argc, char **argv) {

    // Use the most general type to be able to read in all detector files
    using detector_t = detray::detector<>;

    // Filter out the google test flags
    ::testing::InitGoogleTest(&argc, argv);

    // Specific options for this test
    po::options_description desc("\ndetray CUDA material validation options");

    desc.add_options()(
        "tol", boost::program_options::value<float>()->default_value(1.f),
        "Tolerance for comparing the material traces [%]");

    // Configs to be filled
    detray::io::detector_reader_config reader_cfg{};
    detray::test::material_validation<detector_t>::config mat_val_cfg{};
    test::material_scan<detector_t>::config mat_scan_cfg{};

    po::variables_map vm = detray::options::parse_options(
        desc, argc, argv, reader_cfg, mat_scan_cfg.track_generator(),
        mat_val_cfg.propagation());

    // General options
    if (vm.count("tol")) {
        mat_val_cfg.relative_error(vm["tol"].as<float>() / 100.f);
    }

    /// Vecmem memory resource for the device allocations
    vecmem::cuda::device_memory_resource dev_mr{};
    vecmem::host_memory_resource host_mr;

    const auto [det, names] =
        detray::io::read_detector<detector_t>(host_mr, reader_cfg);

    auto white_board = std::make_shared<test::whiteboard>();

    // Print the detector's material as recorded by a ray scan
    mat_val_cfg.name("material_validation_for_cuda");
    mat_scan_cfg.whiteboard(white_board);
    mat_scan_cfg.track_generator().uniform_eta(true);
    detray::detail::register_checks<test::material_scan>(det, names,
                                                         mat_scan_cfg);

    // Now trace the material during navigation and compare
    mat_val_cfg.name("material_validation_cuda");
    mat_val_cfg.whiteboard(white_board);
    mat_val_cfg.device_mr(&dev_mr);

    detail::register_checks<detray::cuda::material_validation>(det, names,
                                                               mat_val_cfg);

    // Run the checks
    return RUN_ALL_TESTS();
}
