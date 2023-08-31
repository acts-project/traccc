/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Boost
#include <boost/program_options.hpp>

namespace traccc {

namespace po = boost::program_options;

struct full_tracking_input_config {
    std::string digitization_config_file;

    full_tracking_input_config(po::options_description& desc);
    void read(const po::variables_map& vm);
};

}  // namespace traccc
