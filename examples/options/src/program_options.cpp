/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/options/program_options.hpp"

// System include(s).
#include <cstdlib>
#include <iostream>

namespace traccc::opts {

program_options::program_options(
    std::string_view description,
    const std::vector<std::reference_wrapper<interface> >& options, int argc,
    char* argv[])
    : m_desc(std::string{description}) {

    // Add all of the option groups.
    for (const interface& opt : options) {
        m_desc.add(opt.options());
    }
    // Add a help option.
    m_desc.add_options()("help,h", "Print this help message");

    // Parse the command line options.
    boost::program_options::variables_map vm;
    boost::program_options::store(
        boost::program_options::parse_command_line(argc, argv, m_desc), vm);

    // Print a help message if the user asked for it.
    if (vm.count("help")) {
        std::cout << m_desc << std::endl;
        std::exit(0);
    }

    // Handle any and all errors.
    try {
        boost::program_options::notify(vm);
    } catch (const std::exception& ex) {
        std::cerr << "Couldn't interpret command line options because of:\n\n"
                  << ex.what() << "\n\n"
                  << m_desc << std::endl;
        std::exit(1);
    }

    // Read / post-process the options.
    for (interface& opt : options) {
        opt.read(vm);
    }

    // Tell the user what's happening.
    std::cout << "\nRunning " << description << "\n\n";
    for (const auto& opt : options) {
        std::cout << opt << "\n";
    }
    std::cout << std::endl;
}

}  // namespace traccc::opts
