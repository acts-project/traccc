/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Boost include(s).
#include <boost/program_options.hpp>

// System include(s).
#include <cstddef>
#include <iosfwd>
#include <string>

namespace traccc::opts {

/// Command line options used in the throughput tests
class throughput {

    public:
    /// @name Options
    /// @{

    /// The number of events to process during the job
    std::size_t processed_events = 100;
    /// The number of events to run "cold", i.e. run without accounting for
    /// them in the performance measurements
    std::size_t cold_run_events = 10;

    /// Output log file
    std::string log_file;

    /// @}

    /// Constructor on top of a common @c program_options object
    ///
    /// @param desc The program options to add to
    ///
    throughput(boost::program_options::options_description& desc);

    /// Read/process the command line options
    ///
    /// @param vm The command line options to interpret/read
    ///
    void read(const boost::program_options::variables_map& vm);

    private:
    /// Description of this program option group
    boost::program_options::options_description m_desc;

};  // class throughput

/// Printout helper for @c traccc::opts::throughput
std::ostream& operator<<(std::ostream& out, const throughput& opt);

}  // namespace traccc::opts
