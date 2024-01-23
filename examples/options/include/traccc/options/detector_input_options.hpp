/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Boost
#include <boost/program_options.hpp>

namespace traccc {

namespace po = boost::program_options;

struct detector_input_options {
    std::string detector_file;
    std::string material_file;
    std::string grid_file;

    detector_input_options(po::options_description& desc);
    void read(const po::variables_map& vm);
};

}  // namespace traccc