/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_candidate_collection.hpp"

namespace traccc::details {

/// @c traccc::is_same_object specialisation for
/// @c traccc::track_candidate_collection_types::host
template <typename T>
class is_same_object<edm::track_candidate<T>> {

    public:
    /// Constructor with a reference object, and an allowed uncertainty
    is_same_object(const measurement_collection_types::const_view& ref_meas,
                   const measurement_collection_types::const_view& test_meas,
                   const edm::track_candidate<T>& ref,
                   scalar unc = float_epsilon)
        : m_ref_meas(ref_meas),
          m_test_meas(test_meas),
          m_ref(ref),
          m_unc(unc) {}

    /// Specialised implementation for @c traccc::bound_track_parameters
    bool operator()(const edm::track_candidate<T>& obj) const {

        // The two track candidates need to have the same parameters.
        if (!(is_same_object<bound_track_parameters<>>(m_ref.params(),
                                                       m_unc)(obj.params()) &&
              is_same_scalar(obj.ndf(), m_ref.ndf(), m_unc))) {
            return false;
        }

        // The two track candidates need to have the same number of
        // measurements.
        if (obj.measurement_indices().size() !=
            m_ref.measurement_indices().size()) {
            return false;
        }

        // Now compare the measurements one by one.
        const measurement_collection_types::const_device ref_meas{m_ref_meas};
        const measurement_collection_types::const_device test_meas{m_test_meas};
        for (unsigned int i = 0; i < obj.measurement_indices().size(); ++i) {
            if (!is_same_object<measurement>(
                    ref_meas.at(m_ref.measurement_indices()[i]),
                    m_unc)(test_meas.at(obj.measurement_indices()[i]))) {
                return false;
            }
        }

        // If we got here, the two track candidates are the same.
        return true;
    }

    private:
    /// Measurements for the reference object
    const measurement_collection_types::const_view m_ref_meas;
    /// Measurements for the test object
    const measurement_collection_types::const_view m_test_meas;
    /// The reference object
    const edm::track_candidate<T> m_ref;
    /// The uncertainty
    scalar m_unc;

};  // class is_same_object<track_candidate_collection_types::host>

}  // namespace traccc::details
