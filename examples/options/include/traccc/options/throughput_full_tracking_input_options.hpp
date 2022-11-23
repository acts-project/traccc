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

// System include(s).
#include <iostream>

namespace traccc {

namespace po = boost::program_options;

struct throughput_full_tracking_input_config {
    traccc::data_format input_data_format = traccc::data_format::csv;
    std::string input_directory;
    std::string detector_file;
    std::string digitization_config_file;
    int loaded_events;
    int processed_events;

    throughput_full_tracking_input_config(po::options_description& desc);
    void read(const po::variables_map& vm);
};

std::ostream& operator<<(std::ostream& out,
                         const throughput_full_tracking_input_config& cfg);

}  // namespace traccc