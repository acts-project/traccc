/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/performance/details/comparator_factory.hpp"

namespace traccc::details {

/// @name Implementation for @c traccc::details::comparator_factory<seed>
/// @{

comparator_factory<alt_seed>::comparator_factory(
    const spacepoint_collection_types::const_view& ref_spacepoints,
    const spacepoint_collection_types::const_view& test_spacepoints)
    : m_ref_spacepoints(ref_spacepoints),
      m_test_spacepoints(test_spacepoints) {}

is_same_object<alt_seed> comparator_factory<alt_seed>::make_comparator(
    const alt_seed& ref, scalar unc) const {

    return is_same_object<alt_seed>(m_ref_spacepoints, m_test_spacepoints, ref,
                                    unc);
}

/// @}

}  // namespace traccc::details
