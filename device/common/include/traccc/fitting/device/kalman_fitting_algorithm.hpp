/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/device/algorithm_base.hpp"
#include "traccc/fitting/device/fit.hpp"

// Project include(s).
#include "traccc/bfield/magnetic_field.hpp"
#include "traccc/edm/device/sort_key.hpp"
#include "traccc/edm/track_container.hpp"
#include "traccc/fitting/details/kalman_fitting_types.hpp"
#include "traccc/fitting/device/fit_prelude.hpp"
#include "traccc/fitting/fitting_config.hpp"
#include "traccc/geometry/detector_buffer.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"
#include "traccc/utils/messaging.hpp"

namespace traccc::device {

/// Kalman filter based track fitting algorithm
class kalman_fitting_algorithm
    : public algorithm<edm::track_container<default_algebra>::buffer(
          const detector_buffer&, const magnetic_field&,
          const edm::track_container<default_algebra>::const_view&)>,
      public messaging,
      public algorithm_base {

    public:
    /// Configuration type
    using config_type = fitting_config;

    /// Constructor with the algorithm's configuration
    ///
    /// @param config The configuration object
    /// @param mr     The memory resource(s) used by the algorithm
    /// @param copy   The copy object used by the algorithm
    /// @param logger The logger used by the algorithm
    ///
    kalman_fitting_algorithm(
        const config_type& config, const traccc::memory_resource& mr,
        const vecmem::copy& copy,
        std::unique_ptr<const Logger> logger = getDummyLogger().clone());
    /// Destructor
    virtual ~kalman_fitting_algorithm();

    /// Operator executing the algorithm.
    ///
    /// @param det              The detector object
    /// @param field            The magnetic field object
    /// @param track_candidates All track candidates to fit
    ///
    /// @return A container of the fitted track states
    ///
    output_type operator()(
        const detector_buffer& det, const magnetic_field& field,
        const edm::track_container<default_algebra>::const_view&
            track_candidates) const override;

    protected:
    /// @name Type(s)/function(s) used internally by the algorithm
    /// @{

    /// Base class for the payload of the fitting kernels
    struct fit_payload_base {
        /// Constructor
        fit_payload_base(const detector_buffer& det,
                         const magnetic_field& field);
        /// Virtual constructor for a polymorphic payload object
        virtual ~fit_payload_base();
        /// Polymorphic tracking geometry buffer
        const detector_buffer& detector;
        /// Polymorphic magnetic field object
        const magnetic_field& field;
    };

    /// Concrete payload type for the fitting kernels
    template <typename detector_t, typename bfield_t>
    struct fit_payload : public fit_payload_base {

        /// Type of the fitter that would use this payload
        using fitter_type =
            traccc::details::kalman_fitter_t<detector_t, bfield_t>;
        /// Type of the payload for the fitting kernels
        using payload_type = device::fit_payload<
            traccc::details::kalman_fitter_t<detector_t, bfield_t>>;

        /// Constructor to help with creating such objects with
        /// @c std::make_unique
        fit_payload(const detector_buffer& det, const magnetic_field& _field,
                    const payload_type& payload)
            : fit_payload_base{det, _field}, host_payload{payload} {}

        /// Surfaces hit by the fitted tracks, used during the fit
        vecmem::data::jagged_vector_buffer<typename detector_t::surface_type>
            surfaces;

        /// The host-resident payload
        payload_type host_payload;
        /// The device-resident payload
        vecmem::data::vector_buffer<payload_type> device_payload;
    };

    /// Prepare a detector+bfield specific payload for the fitting kernel(s)
    ///
    /// Function to be used by the specific @c prepare_fit_payload functions
    /// for preparing the payload. Since apart from different template types,
    /// they all work the same way.
    ///
    /// @tparam detector_list_t The list of supported detector types to use for
    ///                        the visitor
    /// @tparam bfield_list_t  The list of supported b-field types to use for
    ///                        the visitor
    ///
    /// @param det             The detector buffer to prepare the payload for
    /// @param field           The magnetic field to prepare the payload for
    /// @param n_surfaces      The number of surfaces for each track to be
    ///                        fitted
    /// @param track_indices   The fitting order of the tracks
    /// @param track_liveness  The buffer to write the track liveness into
    /// @param tracks          The tracks to be fitted
    ///
    /// @return The prepared payload for the fitting kernel(s)
    ///
    template <typename detector_list_t, typename bfield_list_t>
    std::unique_ptr<fit_payload_base> prepare_fit_payload_helper(
        const detector_buffer& det, const magnetic_field& field,
        const std::vector<unsigned int>& n_surfaces,
        const vecmem::data::vector_view<const unsigned int>& track_indices,
        vecmem::data::vector_view<unsigned int>& track_liveness,
        edm::track_container<default_algebra>::view tracks) const;

    /// Safely cast a base payload to a concrete payload type
    template <typename detector_t, typename bfield_t>
    static const fit_payload<detector_t, bfield_t>& cast_fit_payload(
        const fit_payload_base& payload);

    /// @}

    /// @name Function(s) to be implemented by derived classes
    /// @{

    /// Prepare a buffer with the index order with which to fit the tracks
    ///
    /// @param[in] tracks The tracks to be fitted
    /// @param[out] track_sort_keys Buffer storing temporary sorting keys
    /// @param[out] track_indices The buffer to write the fitting order into
    ///
    virtual void prepare_track_fit_order(
        const edm::track_collection<default_algebra>::const_view& tracks,
        vecmem::data::vector_view<sort_key>& track_sort_keys,
        vecmem::data::vector_view<unsigned int>& track_indices) const = 0;

    /// Kernel to prepare the fitting payloads
    ///
    /// @param payload The payload for the kernel(s)
    ///
    virtual void fit_prelude_kernel(
        const fit_prelude_payload& payload) const = 0;

    /// Function preparing the fitting payload
    ///
    /// @param det             The detector buffer to prepare the payload for
    /// @param field           The magnetic field to prepare the payload for
    /// @param n_surfaces      The number of surfaces for each track to be
    ///                        fitted
    /// @param track_indices   The fitting order of the tracks
    /// @param track_liveness  The buffer to write the track liveness into
    /// @param tracks          The tracks to be fitted
    ///
    /// @return The prepared payload for the fitting kernel(s)
    ///
    virtual std::unique_ptr<fit_payload_base> prepare_fit_payload(
        const detector_buffer& det, const magnetic_field& field,
        const std::vector<unsigned int>& n_surfaces,
        const vecmem::data::vector_view<const unsigned int>& track_indices,
        vecmem::data::vector_view<unsigned int>& track_liveness,
        edm::track_container<default_algebra>::view tracks) const = 0;

    /// Function launching the "forward fitting" kernel(s)
    ///
    /// @param config The fitting configuration
    /// @param payload The payload for the fitting kernel(s)
    ///
    virtual void fit_forward_kernel(const fitting_config& config,
                                    const fit_payload_base& payload) const = 0;

    /// Function launching the "backward fitting" kernel(s)
    ///
    /// @param config The fitting configuration
    /// @param payload The payload for the fitting kernel(s)
    ///
    virtual void fit_backward_kernel(const fitting_config& config,
                                     const fit_payload_base& payload) const = 0;

    /// @}

    private:
    /// Internal data type
    struct data;
    /// Pointer to internal data
    std::unique_ptr<data> m_data;

};  // class kalman_fitting_algorithm

}  // namespace traccc::device

// Include the implementation.
#include "traccc/fitting/device/impl/kalman_fitting_algorithm.ipp"
