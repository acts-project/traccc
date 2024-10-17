/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_detector_description.hpp"
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

    // Memory resource used by the EDM.
    vecmem::host_memory_resource host_mr;

    // Construct the detector description object.
    traccc::silicon_detector_description::host det_descr{host_mr};
    traccc::io::read_detector_description(
        det_descr, detector_opts.detector_file, detector_opts.digitization_file,
        (detector_opts.use_detray_detector ? traccc::data_format::json
                                           : traccc::data_format::csv));

    // Loop over events
    for (unsigned int event = input_opts.skip;
         event < input_opts.events + input_opts.skip; ++event) {

        // Read the cells from the relevant event file
        traccc::edm::silicon_cell_collection::host cells{host_mr};
        traccc::io::read_cells(cells, event, input_opts.directory, &det_descr,
                               input_opts.format);

        // Write binary file
        traccc::io::write(event, output_opts.directory,
                          traccc::data_format::binary, vecmem::get_data(cells));

        // Read the hits from the relevant event file
        traccc::spacepoint_collection_types::host spacepoints{&host_mr};
        traccc::io::read_spacepoints(spacepoints, event, input_opts.directory,
                                     nullptr, input_opts.format);

        // Write binary file
        traccc::io::write(event, output_opts.directory,
                          traccc::data_format::binary,
                          vecmem::get_data(spacepoints));

        // Read the measurements from the relevant event file
        traccc::measurement_collection_types::host measurements{&host_mr};
        traccc::io::read_measurements(measurements, event, input_opts.directory,
                                      nullptr, input_opts.format);

        // Write binary file
        traccc::io::write(event, output_opts.directory,
                          traccc::data_format::binary,
                          vecmem::get_data(measurements));
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
