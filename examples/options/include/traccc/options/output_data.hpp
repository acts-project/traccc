/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/io/data_format.hpp"

// Boost include(s).
#include <boost/program_options.hpp>

// System include(s).
#include <iosfwd>
#include <string>

namespace traccc::opts {

/// Options for the output data that a given application would produce
class output_data {

    public:
    /// @name Options
    /// @{

    /// The data format of the input files
    traccc::data_format format = data_format::csv;
    /// Directory of the input files
    std::string directory = "testing/";

    /// @}

    /// Constructor on top of a common @c program_options object
    ///
    /// @param desc The program options to add to
    ///
    output_data(boost::program_options::options_description& desc);

    /// Read/process the command line options
    ///
    /// @param vm The command line options to interpret/read
    ///
    void read(const boost::program_options::variables_map& vm);

    private:
    /// Description of this program option group
    boost::program_options::options_description m_desc;

};  // class output_data

/// Printout helper for @c traccc::opts::output_data
std::ostream& operator<<(std::ostream& out, const output_data& opt);

}  // namespace traccc::opts
