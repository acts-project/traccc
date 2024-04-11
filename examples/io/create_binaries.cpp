/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_digitization_config.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/read_measurements.hpp"
#include "traccc/io/read_spacepoints.hpp"
#include "traccc/io/write.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/output_data.hpp"
#include "traccc/options/program_options.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// System include(s).
#include <cstdlib>

int create_binaries(const traccc::opts::detector& detector_opts,
                    const traccc::opts::input_data& input_opts,
                    const traccc::opts::output_data& output_opts) {

    // Read the surface transforms
    auto [surface_transforms, _] =
        traccc::io::read_geometry(detector_opts.detector_file);

    // Read the digitization configuration file
    auto digi_cfg =
        traccc::io::read_digitization_config(detector_opts.digitization_file);

    // Memory resource used by the EDM.
    vecmem::host_memory_resource host_mr;

    // Loop over events
    for (unsigned int event = input_opts.skip;
         event < input_opts.events + input_opts.skip; ++event) {

        // Read the cells from the relevant event file
        traccc::io::cell_reader_output cells_csv(&host_mr);
        traccc::io::read_cells(cells_csv, event, input_opts.directory,
                               input_opts.format, &surface_transforms,
                               &digi_cfg);

        // Write binary file
        traccc::io::write(event, output_opts.directory,
                          traccc::data_format::binary,
                          vecmem::get_data(cells_csv.cells),
                          vecmem::get_data(cells_csv.modules));

        // Read the hits from the relevant event file
        traccc::io::spacepoint_reader_output spacepoints_csv(&host_mr);
        traccc::io::read_spacepoints(spacepoints_csv, event,
                                     input_opts.directory, surface_transforms,
                                     input_opts.format);

        // Write binary file
        traccc::io::write(event, output_opts.directory,
                          traccc::data_format::binary,
                          vecmem::get_data(spacepoints_csv.spacepoints),
                          vecmem::get_data(spacepoints_csv.modules));

        // Read the measurements from the relevant event file
        traccc::io::measurement_reader_output measurements_csv(&host_mr);
        traccc::io::read_measurements(measurements_csv, event,
                                      input_opts.directory, input_opts.format);

        // Write binary file
        traccc::io::write(event, output_opts.directory,
                          traccc::data_format::binary,
                          vecmem::get_data(measurements_csv.measurements),
                          vecmem::get_data(measurements_csv.modules));
    }

    return EXIT_SUCCESS;
}

// The main routine
//
int main(int argc, char* argv[]) {

    // Program options.
    traccc::opts::detector detector_opts;
    traccc::opts::input_data input_opts;
    traccc::opts::output_data output_opts;
    traccc::opts::program_options program_opts{
        "Binary File Creation",
        {detector_opts, input_opts, output_opts},
        argc,
        argv};

    // Run the application.
    return create_binaries(detector_opts, input_opts, output_opts);
}
