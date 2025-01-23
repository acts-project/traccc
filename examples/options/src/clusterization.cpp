/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/options/clusterization.hpp"

#include "traccc/clusterization/clusterization_algorithm.hpp"
#include "traccc/examples/utils/printable.hpp"

// System include(s).
#include <iostream>

namespace traccc::opts {

clusterization::clusterization() : interface("Clusterization Options") {

    m_desc.add_options()("threads-per-partition",
                         boost::program_options::value(&threads_per_partition)
                             ->default_value(256),
                         "The number of threads per partition");
    m_desc.add_options()(
        "max-cells-per-thread",
        boost::program_options::value(&max_cells_per_thread)->default_value(16),
        "The maximum number of cells per thread");
    m_desc.add_options()("target-cells-per-thread",
                         boost::program_options::value(&target_cells_per_thread)
                             ->default_value(8),
                         "The target number of cells per thread");
    m_desc.add_options()("backup-size-multiplier",
                         boost::program_options::value(&backup_size_multiplier)
                             ->default_value(256),
                         "The size multiplier of the backup scratch space");
}

clusterization::operator clustering_config() const {
    clustering_config rv;

    rv.threads_per_partition = threads_per_partition;
    rv.max_cells_per_thread = max_cells_per_thread;
    rv.target_cells_per_thread = target_cells_per_thread;
    rv.backup_size_multiplier = backup_size_multiplier;

    return rv;
}

clusterization::operator host::clusterization_algorithm::config_type() const {
    return {};
}

std::unique_ptr<configuration_printable> clusterization::as_printable() const {
    std::unique_ptr<configuration_printable> cat =
        std::make_unique<configuration_category>("Clustering options");

    dynamic_cast<configuration_category &>(*cat).add_child(
        std::make_unique<configuration_kv_pair>(
            "Threads per partition", std::to_string(threads_per_partition)));
    dynamic_cast<configuration_category &>(*cat).add_child(
        std::make_unique<configuration_kv_pair>(
            "Target cells per thread",
            std::to_string(target_cells_per_thread)));
    dynamic_cast<configuration_category &>(*cat).add_child(
        std::make_unique<configuration_kv_pair>(
            "Max cells per thread", std::to_string(max_cells_per_thread)));
    dynamic_cast<configuration_category &>(*cat).add_child(
        std::make_unique<configuration_kv_pair>(
            "Scratch space multiplier",
            std::to_string(backup_size_multiplier)));

    return cat;
}
}  // namespace traccc::opts
