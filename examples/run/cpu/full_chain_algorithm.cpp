/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "full_chain_algorithm.hpp"

namespace traccc {

full_chain_algorithm::full_chain_algorithm(vecmem::memory_resource& mr)
    : m_clusterization(mr),
      m_spacepoint_formation(mr),
      m_seeding(mr),
      m_track_parameter_estimation(mr) {}

full_chain_algorithm::output_type full_chain_algorithm::operator()(
    const cell_container_types::host& cells) const {

    const spacepoint_formation::output_type spacepoints =
        m_spacepoint_formation(m_clusterization(cells));
    return m_track_parameter_estimation(spacepoints, m_seeding(spacepoints));
}

}  // namespace traccc
