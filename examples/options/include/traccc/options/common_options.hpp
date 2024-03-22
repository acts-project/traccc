/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "traccc/io/data_format.hpp"

// Boost include(s).
#include <boost/program_options.hpp>

// System include(s).
#include <iosfwd>
#include <string>

namespace traccc {

/// Common options for the example applications
struct common_options {

    /// The input data format
    traccc::data_format input_data_format = traccc::data_format::csv;
    /// The data input directory
    std::string input_directory = "tml_full/ttbar_mu20/";
    /// The number of events to process
    unsigned int events = 1;
    /// The number of events to skip
    unsigned int skip = 0;
    /// The number of cells to merge in a partition
    unsigned short target_cells_per_partition = 1024;
    /// Whether to check the reconstructions performance
    bool check_performance = false;
    /// Whether to perform ambiguity resolution
    bool perform_ambiguity_resolution = true;

    /// Constructor on top of a common @c program_options object
    ///
    /// @param desc The program options to add to
    ///
    common_options(boost::program_options::options_description& desc);

    /// Read/process the command line options
    ///
    /// @param vm The command line options to interpret/read
    ///
    void read(const boost::program_options::variables_map& vm);

};  // struct common_options

/// Printout helper for @c traccc::common_options
std::ostream& operator<<(std::ostream& out, const common_options& opt);

}  // namespace traccc
