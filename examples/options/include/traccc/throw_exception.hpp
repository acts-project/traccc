/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Boost
#include <boost/program_options.hpp>

namespace po = boost::program_options;

namespace traccc {

enum exception : int {
    print_help = 0,
    input_error = 1,
    no_exception = 2,
};

int throw_exception(const po::options_description& desc,
                    po::variables_map& vm) {

    // Print a help message if the user asked for it.
    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return print_help;
    }

    // Handle any and all errors.
    try {
        po::notify(vm);
    } catch (const std::exception& ex) {
        std::cerr << "Couldn't interpret command line options because of:\n\n"
                  << ex.what() << "\n\n"
                  << desc << std::endl;
        return input_error;
    }

    return no_exception;
}

}  // namespace traccc