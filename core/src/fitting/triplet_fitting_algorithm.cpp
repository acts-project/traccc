/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/fitting/triplet_fitting_algorithm.hpp"

namespace traccc::host {

triplet_fitting_algorithm::triplet_fitting_algorithm(
    const config_type& config, vecmem::memory_resource& mr, vecmem::copy& copy,
    std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)), m_config{config}, m_mr{mr}, m_copy(copy) {}

}  // namespace traccc::host
