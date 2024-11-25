/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/clusterization/clusterization_algorithm.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/silicon_cell_collection.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/geometry/pixel_data.hpp"
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_detector.hpp"
#include "traccc/io/read_detector_description.hpp"
#include "traccc/options/clusterization.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/program_options.hpp"
#include "traccc/seeding/silicon_pixel_spacepoint_formation_algorithm.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// Boost
#include <boost/program_options.hpp>

// OpenMP
#ifdef _OPENMP
#include "omp.h"
#endif

// System include(s).
#include <chrono>
#include <exception>
#include <iostream>

namespace po = boost::program_options;

int par_run(const traccc::opts::input_data& input_opts,
            const traccc::opts::detector& detector_opts,
            const traccc::opts::clusterization& /*clusterization_opts*/) {

    // Memory resource used by the EDM.
    vecmem::host_memory_resource resource;

    // Construct the detector description object.
    traccc::silicon_detector_description::host det_descr{resource};
    traccc::io::read_detector_description(
        det_descr, detector_opts.detector_file, detector_opts.digitization_file,
        (detector_opts.use_detray_detector ? traccc::data_format::json
                                           : traccc::data_format::csv));
    traccc::silicon_detector_description::data det_descr_data{
        vecmem::get_data(det_descr)};

    // Construct a Detray detector object, if supported by the configuration.
    traccc::default_detector::host detector{resource};
    if (detector_opts.use_detray_detector) {
        traccc::io::read_detector(
            detector, resource, detector_opts.detector_file,
            detector_opts.material_file, detector_opts.grid_file);
    }

    // Type definitions
    using spacepoint_formation_algorithm =
        traccc::host::silicon_pixel_spacepoint_formation_algorithm;

    // Algorithms
    traccc::host::clusterization_algorithm ca(resource);
    spacepoint_formation_algorithm sf(resource);

    // Output stats
    uint64_t n_cells = 0;
    uint64_t n_measurements = 0;
    uint64_t n_spacepoints = 0;

#pragma omp parallel for reduction(+ : n_cells, n_measurements, n_spacepoints)
    // Loop over events
    for (std::size_t event = input_opts.skip;
         event < input_opts.events + input_opts.skip; ++event) {

        // Read the cells from the relevant event file
        traccc::edm::silicon_cell_collection::host cells_per_event{resource};
        static constexpr bool DEDUPLICATE = true;
        traccc::io::read_cells(cells_per_event, event, input_opts.directory,
                               &det_descr, input_opts.format, DEDUPLICATE,
                               input_opts.use_acts_geom_source);

        /*-------------------
            Clusterization
          -------------------*/

        auto measurements_per_event =
            ca(vecmem::get_data(cells_per_event), det_descr_data);

        /*------------------------
            Spacepoint formation
          ------------------------*/

        auto spacepoints_per_event =
            sf(detector, vecmem::get_data(measurements_per_event));

        /*----------------------------
          Statistics
          ----------------------------*/

        n_cells += cells_per_event.size();
        n_measurements += measurements_per_event.size();
        n_spacepoints += spacepoints_per_event.size();
    }

#pragma omp critical

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- read    " << n_cells << " cells" << std::endl;
    std::cout << "- created " << n_measurements << " measurements. "
              << std::endl;
    std::cout << "- created " << n_spacepoints << " spacepoints. " << std::endl;

    return 0;
}

// The main routine
//
int main(int argc, char* argv[]) {

    // Program options.
    traccc::opts::detector detector_opts;
    traccc::opts::input_data input_opts;
    traccc::opts::clusterization clusterization_opts;
    traccc::opts::program_options program_opts{
        "Clusterization + Spacepoint Formation with OpenMP",
        {detector_opts, input_opts, clusterization_opts},
        argc,
        argv};

    auto start = std::chrono::system_clock::now();
    const int result = par_run(input_opts, detector_opts, clusterization_opts);
    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> diff = end - start;
    std::cout << "Execution time: " << diff.count() << " sec." << std::endl;
    return result;
}
