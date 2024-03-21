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
#include <iosfwd>

namespace traccc::opts {

/// Options for the cell clusterization algorithm(s)
class clusterization {

    public:
    /// @name Options
    /// @{

    /// The number of cells to merge in a partition
    unsigned short target_cells_per_partition = 1024;

    /// @}

    /// Constructor on top of a common @c program_options object
    ///
    /// @param desc The program options to add to
    ///
    clusterization(boost::program_options::options_description& desc);

    /// Read/process the command line options
    ///
    /// @param vm The command line options to interpret/read
    ///
    void read(const boost::program_options::variables_map& vm);

    private:
    /// Description of this program option group
    boost::program_options::options_description m_desc;

};  // class clusterization

/// Printout helper for @c traccc::opts::clusterization
std::ostream& operator<<(std::ostream& out, const clusterization& opt);

}  // namespace traccc::opts
