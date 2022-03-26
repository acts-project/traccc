/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Boost
#include <boost/program_options.hpp>

// STD
#include <iostream>

namespace traccc {

namespace po = boost::program_options;

void handle_argument_errors(po::variables_map& vm,
                            const po::options_description& desc);

}  // namespace traccc