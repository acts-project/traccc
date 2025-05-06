/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/edm/track_parameters.hpp"
#include "traccc/fitting/kalman_filter/kalman_actor.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/io/data_format.hpp"
#include "traccc/utils/event_data.hpp"
#include "traccc/utils/logging.hpp"
#include "traccc/utils/propagation.hpp"

// Options
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/track_propagation.hpp"

// Performance include(s).
#include "traccc/performance/navigation_comparison.hpp"

// Detray include(s)
#include <detray/io/frontend/detector_reader.hpp>

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// The main routine
//
int main(int argc, char* argv[]) {
    std::unique_ptr<const traccc::Logger> logger = traccc::getDefaultLogger(
        "TracccExampleNavValCPU", traccc::Logging::Level::INFO);

    // Program options.
    traccc::opts::detector detector_opts;
    traccc::opts::input_data input_opts;
    traccc::opts::track_propagation propagation_opts;
    traccc::opts::program_options program_opts{
        "Navigation validation on the Host",
        {detector_opts, input_opts, propagation_opts},
        argc,
        argv,
        logger->cloneWithSuffix("Options")};

    // Memory resource used by the application.
    vecmem::host_memory_resource host_mr;

    // Set up the detector reader configuration.
    detray::io::detector_reader_config reader_cfg;
    cfg.add_file(traccc::io::get_absolute_path(detector_opts.detector_file));
    if (detector_opts.material_file.empty() == false) {
        cfg.add_file(
            traccc::io::get_absolute_path(detector_opts.material_file));
    }
    if (detector_opts.grid_file.empty() == false) {
        cfg.add_file(traccc::io::get_absolute_path(detector_opts.grid_file));
    }

    // Read the detector.
    auto [det, names] =
        detray::io::read_detector<detector_t>(host_mr, reader_cfg);

    // Run the application.
    return navigation_comparison(det, names, propagation_opts.m_config,
                                 input_opts.directory, input_dir.events,
                                 logger->clone());
}
