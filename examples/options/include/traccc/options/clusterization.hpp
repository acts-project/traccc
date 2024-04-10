/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/options/details/interface.hpp"

namespace traccc::opts {

/// Options for the cell clusterization algorithm(s)
class clusterization : public interface {

    public:
    /// @name Options
    /// @{

    /// The number of cells to merge in a partition
    unsigned short target_cells_per_partition = 1024;

    /// @}

    /// Constructor
    clusterization();

    private:
    /// Print the specific options of this class
    std::ostream& print_impl(std::ostream& out) const override;

};  // class clusterization

}  // namespace traccc::opts
