/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Boost include(s).
#include <boost/program_options.hpp>

// System include(s).
#include <cstddef>
#include <iosfwd>

namespace traccc::opts {

/// Option(s) for accelerator usage
class accelerator {

    public:
    /// @name Options
    /// @{

    /// Whether to compare the accelerator code's output with that of the CPU
    bool compare_with_cpu = false;

    /// @}

    /// Constructor on top of a common @c program_options object
    ///
    /// @param desc The program options to add to
    ///
    accelerator(boost::program_options::options_description& desc);

    /// Read/process the command line options
    ///
    /// @param vm The command line options to interpret/read
    ///
    void read(const boost::program_options::variables_map& vm);

    private:
    /// Description of this program option group
    boost::program_options::options_description m_desc;

};  // struct accelerator

/// Printout helper for @c traccc::opts::accelerator
std::ostream& operator<<(std::ostream& out, const accelerator& opt);

}  // namespace traccc::opts
