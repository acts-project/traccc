/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/clusterization/clustering_config.hpp"
#include "traccc/clusterization/clusterization_algorithm.hpp"
#include "traccc/options/details/config_provider.hpp"
#include "traccc/options/details/interface.hpp"

namespace traccc::opts {

/// Options for the cell clusterization algorithm(s)
class clusterization
    : public interface,
      public config_provider<clustering_config>,
      public config_provider<host::clusterization_algorithm::config_type> {

    public:
    /// Constructor
    clusterization();

    /// Configuration conversion
    operator clustering_config() const override;
    operator host::clusterization_algorithm::config_type() const override;

    private:
    /// @name Options
    /// @{
    /// The number of cells to merge in a partition
    unsigned int threads_per_partition;
    unsigned int max_cells_per_thread;
    unsigned int target_cells_per_thread;
    unsigned int backup_size_multiplier;
    /// @}

    /// Print the specific options of this class
    std::ostream& print_impl(std::ostream& out) const override;

};  // class clusterization

}  // namespace traccc::opts
