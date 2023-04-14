/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/seed.hpp"
#include "traccc/edm/spacepoint.hpp"

namespace traccc::details {

/// @c traccc::details::comparator_factory specialisation for @c traccc::seed
template <>
class comparator_factory<seed> {

    public:
    /// Constructor with all necessary arguments
    comparator_factory(
        const spacepoint_collection_types::const_view& ref_spacepoints,
        const spacepoint_collection_types::const_view& test_spacepoints);

    /// Instantiate an instance of a comparator object
    is_same_object<seed> make_comparator(const seed& ref,
                                         scalar unc = float_epsilon) const;

    private:
    /// Spacepoint container for the reference seeds
    const spacepoint_collection_types::const_view m_ref_spacepoints;
    /// Spacepoint container for the test seeds
    const spacepoint_collection_types::const_view m_test_spacepoints;

};  // class comparator_factory

}  // namespace traccc::details
