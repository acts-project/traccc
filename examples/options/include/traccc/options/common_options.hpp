/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "traccc/io/data_format.hpp"

// Boost
#include <boost/program_options.hpp>

namespace traccc {

namespace po = boost::program_options;

struct common_options {
    traccc::data_format input_data_format = traccc::data_format::csv;
    std::string input_directory;
    unsigned int events;
    int skip;
    unsigned short target_cells_per_partition;

    common_options(po::options_description& desc);
    void read(const po::variables_map& vm);
};

}  // namespace traccc