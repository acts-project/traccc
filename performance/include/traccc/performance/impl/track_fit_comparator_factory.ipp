/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_fit_collection.hpp"
#include "traccc/edm/track_state_collection.hpp"

// Local include(s).
#include "traccc/performance/details/is_same_object.hpp"

namespace traccc::details {

/// @c traccc::details::comparator_factory specialisation for
/// @c traccc::edm::track_fit
template <typename T>
class comparator_factory<edm::track_fit<T>> {

    public:
    /// Constructor with all necessary arguments
    comparator_factory(
        const measurement_collection_types::const_view& ref_meas,
        const measurement_collection_types::const_view& test_meas,
        const edm::track_state_collection<default_algebra>::const_view&
            ref_states,
        const edm::track_state_collection<default_algebra>::const_view&
            test_states)
        : m_ref_meas(ref_meas),
          m_test_meas(test_meas),
          m_ref_states(ref_states),
          m_test_states(test_states) {}

    /// Instantiate an instance of a comparator object
    is_same_object<edm::track_fit<T>> make_comparator(
        const edm::track_fit<T>& ref, scalar unc = float_epsilon) const {

        return is_same_object<edm::track_fit<T>>(
            m_ref_meas, m_test_meas, m_ref_states, m_test_states, ref, unc);
    }

    private:
    /// Measurement container for the reference track candidates
    const measurement_collection_types::const_view m_ref_meas;
    /// Measurement container for the test track candidates
    const measurement_collection_types::const_view m_test_meas;
    /// States for the reference object
    const edm::track_state_collection<default_algebra>::const_view m_ref_states;
    /// States for the test object
    const edm::track_state_collection<default_algebra>::const_view
        m_test_states;

};  // class comparator_factory

}  // namespace traccc::details
