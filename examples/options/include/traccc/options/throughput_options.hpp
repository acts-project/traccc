/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/io/data_format.hpp"

// Boost include(s).
#include <boost/program_options.hpp>

// System include(s).
#include <cstddef>
#include <iosfwd>
#include <string>

namespace traccc {

/// Command line options used in the throughput tests
struct throughput_options {

    /// The data format of the input files
    data_format input_data_format = data_format::csv;
    /// Directory of the input files
    std::string input_directory;
    /// The file describing the detector geometry
    std::string detector_file;
    /// The file describing the detector digitization configuration
    std::string digitization_config_file;

    /// The average number of cells in each partition.
    /// Equal to the number of threads in the clusterization kernels multiplied
    /// by CELLS_PER_THREAD defined in clusterization. Adapt to different GPUs'
    /// capabilities.
    unsigned short target_cells_per_partition;

    /// The number of input events to load into memory
    std::size_t loaded_events = 10;
    /// The number of events to process during the job
    std::size_t processed_events = 100;
    /// The number of events to run "cold", i.e. run without accounting for
    /// them in the performance measurements
    std::size_t cold_run_events = 10;

    /// Output log file
    std::string log_file;

    /// Constructor on top of a common @c program_options object
    ///
    /// @param desc The program options to add to
    ///
    throughput_options(boost::program_options::options_description& desc);

    /// Read the command line options
    ///
    /// @param vm The command line options to interpret/read
    ///
    void read(const boost::program_options::variables_map& vm);

};  // struct throughput_options

/// Printout helper for @c traccc::throughput_options
std::ostream& operator<<(std::ostream& out, const throughput_options& opt);

}  // namespace traccc
