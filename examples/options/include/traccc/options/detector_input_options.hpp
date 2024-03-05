/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Boost include(s).
#include <boost/program_options.hpp>

// System include(s).
#include <iosfwd>
#include <string>

namespace traccc {

/// Options for the detector description
struct detector_input_options {

    /// The file containing the detector description
    std::string detector_file = "tml_detector/trackml-detector.csv";
    /// The file containing the material description
    std::string material_file;
    /// The file containing the surface grid description
    std::string grid_file;

    /// Constructor on top of a common @c program_options object
    ///
    /// @param desc The program options to add to
    ///
    detector_input_options(boost::program_options::options_description& desc);

    /// Read/process the command line options
    ///
    /// @param vm The command line options to interpret/read
    ///
    void read(const boost::program_options::variables_map& vm);

};  // struct detector_input_options

/// Printout helper for @c traccc::detector_input_options
std::ostream& operator<<(std::ostream& out, const detector_input_options& opt);

}  // namespace traccc
