/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Boost
#include "detray/options/boost_program_options.hpp"

// System include(s)
#include <iostream>

namespace detray::options {

/// Add options to the boost options description according to the type T
template <typename T>
void add_options(boost::program_options::options_description &,
                 const T &) { /* Do nothing */
}

/// Fill the configuration type T from the boost variable map
template <typename T>
void configure_options(boost::program_options::variables_map &,
                       T &) { /* Do nothing */
}

/// Print the configuration
template <typename T>
void print_options(T &cfg) {
    std::cout << cfg << std::endl;
}

}  // namespace detray::options
