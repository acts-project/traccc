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

/// Options for the input data that a given application would use
class input_data {

    public:
    /// @name Options
    /// @{

    /// The data format of the input files
    traccc::data_format format = data_format::csv;
    /// Directory of the input files
    std::string directory = "tml_full/ttbar_mu20/";
    /// The number of events to process
    std::size_t events = 1;
    /// The number of events to skip
    std::size_t skip = 0;

    /// @}

    /// Constructor on top of a common @c program_options object
    ///
    /// @param desc The program options to add to
    ///
    input_data(boost::program_options::options_description& desc);

    /// Read/process the command line options
    ///
    /// @param vm The command line options to interpret/read
    ///
    void read(const boost::program_options::variables_map& vm);

    private:
    /// Description of this program option group
    boost::program_options::options_description m_desc;

};  // class input_data

/// Printout helper for @c traccc::opts::input_data
std::ostream& operator<<(std::ostream& out, const input_data& opt);

}  // namespace traccc::opts
