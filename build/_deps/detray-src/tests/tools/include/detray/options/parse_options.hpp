/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Boost
#include "detray/options/boost_program_options.hpp"

// System include(s).
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace detray::options {

// Forward declare the options handling for different configuration types
template <typename T>
void add_options(boost::program_options::options_description&, const T&);

template <typename T>
void configure_options(boost::program_options::variables_map&, T&);

/// Parse commandline options and add them to detray configuration types
template <typename... CONFIGS>
auto parse_options(boost::program_options::options_description& desc, int argc,
                   char* argv[], CONFIGS&... cfgs) {

    static_assert(sizeof...(CONFIGS) > 0, "No commandline options configured");

    if (!argv) {
        throw std::invalid_argument("Invalid command line arguments passed");
    }

    desc.add_options()("help", "Produce help message");

    // Add options according to the configurations that were passed
    (add_options(desc, cfgs), ...);

    // Parse options
    boost::program_options::variables_map vm;
    try {
        boost::program_options::store(
            parse_command_line(
                argc, argv, desc,
                boost::program_options::command_line_style::unix_style ^
                    boost::program_options::command_line_style::allow_short),
            vm);

        boost::program_options::notify(vm);
    } catch (const std::exception& ex) {
        // Print help message in case of error
        std::cout << ex.what() << "\n" << desc << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Print help message when requested
    if (vm.count("help")) {
        std::cout << desc << std::endl;
        std::exit(EXIT_SUCCESS);
    }

    // Add the options to the configurations
    (configure_options(vm, cfgs), ...);
    // Make sure everything is configured correctly
    (print_options(cfgs), ...);

    return vm;
}

/// Parse commandline options and add them to detray configuration types
template <typename... CONFIGS>
auto parse_options(const std::string& description, int argc, char* argv[],
                   CONFIGS&... cfgs) {
    // Options description
    boost::program_options::options_description desc(description);

    // Run options parsing
    return parse_options(desc, argc, argv, cfgs...);
}

}  // namespace detray::options
