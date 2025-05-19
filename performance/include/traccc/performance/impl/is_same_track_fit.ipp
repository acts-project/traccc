/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_fit_collection.hpp"
#include "traccc/edm/track_state_collection.hpp"

// Local include(s).
#include "traccc/performance/impl/is_same_track_state.ipp"

namespace traccc::details {

/// @c traccc::is_same_object specialisation for @c traccc::edm::track_fit<T>
template <typename T>
class is_same_object<edm::track_fit<T>> {

    public:
    /// Constructor with a reference object, and an allowed uncertainty
    is_same_object(
        const measurement_collection_types::const_view& ref_meas,
        const measurement_collection_types::const_view& test_meas,
        const edm::track_state_collection<default_algebra>::const_view&
            ref_states,
        const edm::track_state_collection<default_algebra>::const_view&
            test_states,
        const edm::track_fit<T>& ref, scalar unc = float_epsilon)
        : m_ref_meas(ref_meas),
          m_test_meas(test_meas),
          m_ref_states(ref_states),
          m_test_states(test_states),
          m_ref(ref),
          m_unc(unc) {}

    /// Specialised implementation for @c traccc::edm::track_fit<T>
    bool operator()(const edm::track_fit<T>& obj) const {

        // Compare the fit outcomes.
        if (obj.fit_outcome() != m_ref.fit_outcome()) {
            return false;
        }
        // Compare the parameters.
        if (!is_same_object<bound_track_parameters<>>(m_ref.params(),
                                                      m_unc)(obj.params())) {
            return false;
        }
        // Compare the scalar values.
        if (!is_same_scalar(obj.ndf(), m_ref.ndf(), m_unc) ||
            !is_same_scalar(obj.chi2(), m_ref.chi2(), m_unc) ||
            !is_same_scalar(obj.pval(), m_ref.pval(), m_unc)) {
            return false;
        }
        // Compare the number of holes.
        if (obj.nholes() != m_ref.nholes()) {
            return false;
        }

        // The two tracks need to have the same number of states.
        if (obj.state_indices().size() != m_ref.state_indices().size()) {
            return false;
        }

        // Now compare the track states one by one.
        const edm::track_state_collection<default_algebra>::const_device
            ref_states{m_ref_states};
        const edm::track_state_collection<default_algebra>::const_device
            test_states{m_test_states};
        for (unsigned int i = 0; i < obj.state_indices().size(); ++i) {
            if (!is_same_object<edm::track_state_collection<
                    default_algebra>::const_device::const_proxy_type>(
                    m_ref_meas, m_test_meas,
                    ref_states.at(m_ref.state_indices()[i]),
                    m_unc)(test_states.at(obj.state_indices()[i]))) {
                return false;
            }
        }

        // If we got here, the two track fits are the same.
        return true;
    }

    private:
    /// Measurements for the reference object
    const measurement_collection_types::const_view m_ref_meas;
    /// Measurements for the test object
    const measurement_collection_types::const_view m_test_meas;
    /// States for the reference object
    const edm::track_state_collection<default_algebra>::const_view m_ref_states;
    /// States for the test object
    const edm::track_state_collection<default_algebra>::const_view
        m_test_states;
    /// The reference object
    const edm::track_fit<T> m_ref;
    /// The uncertainty
    scalar m_unc;

};  // class is_same_object<edm::track_fit<T>>

}  // namespace traccc::details
