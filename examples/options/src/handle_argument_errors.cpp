/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// options
#include "traccc/options/handle_argument_errors.hpp"

void traccc::handle_argument_errors(
    traccc::po::variables_map& vm,
    const traccc::po::options_description& desc) {

    // Print a help message if the user asked for it.
    if (vm.count("help")) {
        std::cout << desc << std::endl;
        std::exit(0);
    }

    // Handle any and all errors.
    try {
        po::notify(vm);
    } catch (const std::exception& ex) {
        std::cerr << "Couldn't interpret command line options because of:\n\n"
                  << ex.what() << "\n\n"
                  << desc << std::endl;
        std::exit(1);
    }
}
