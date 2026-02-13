/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/device/algorithm_base.hpp"
#include "traccc/edm/device/sort_key.hpp"
#include "traccc/finding/device/apply_interaction.hpp"
#include "traccc/finding/device/build_tracks.hpp"
#include "traccc/finding/device/fill_finding_duplicate_removal_sort_keys.hpp"
#include "traccc/finding/device/fill_finding_propagation_sort_keys.hpp"
#include "traccc/finding/device/gather_best_tips_per_measurement.hpp"
#include "traccc/finding/device/gather_measurement_votes.hpp"
#include "traccc/finding/device/remove_duplicates.hpp"
#include "traccc/finding/device/update_tip_length_buffer.hpp"

// Project include(s).
#include "traccc/bfield/magnetic_field.hpp"
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/track_container.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/finding/candidate_link.hpp"
#include "traccc/finding/finding_config.hpp"
#include "traccc/geometry/detector_buffer.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"
#include "traccc/utils/messaging.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_buffer.hpp>

// System include(s).
#include <memory>

namespace traccc::device {

/// CKF track finding algorithm
class combinatorial_kalman_filter_algorithm
    : public algorithm<edm::track_container<default_algebra>::buffer(
          const detector_buffer&, const magnetic_field&,
          const edm::measurement_collection<default_algebra>::const_view&,
          const bound_track_parameters_collection_types::const_view&)>,
      public messaging,
      public algorithm_base {

    public:
    /// Configuration type
    using config_type = finding_config;

    /// Constructor with the algorithm's configuration
    ///
    /// @param config The (finding) algorithm configuration
    /// @param mr The memory resource(s) to use in the algorithm
    /// @param copy The copy object to use for copying data between device
    ///             and host memory blocks
    /// @param logger The logger instance to use
    ///
    combinatorial_kalman_filter_algorithm(
        const finding_config& config, const traccc::memory_resource& mr,
        vecmem::copy& copy,
        std::unique_ptr<const Logger> logger = getDummyLogger().clone());
    /// Destructor
    virtual ~combinatorial_kalman_filter_algorithm();

    /// Operator executing the algorithm.
    ///
    /// @param det          The detector object
    /// @param bfield       The magnetic field object
    /// @param measurements All measurements in an event
    /// @param seeds        All seeds in an event to start the track finding
    ///                     with
    /// @return A container of the found tracks
    ///
    output_type operator()(
        const detector_buffer& det, const magnetic_field& bfield,
        const edm::measurement_collection<default_algebra>::const_view&
            measurements,
        const bound_track_parameters_collection_types::const_view& seeds)
        const override;

    protected:
    /// @name Function(s) to be implemented by derived classes
    /// @{

    /// Function meant to perform sanity checks on the input data
    ///
    /// @param measurements All measurements in an event
    /// @return @c true if the input data is valid, @c false otherwise
    ///
    virtual bool input_is_valid(
        const edm::measurement_collection<default_algebra>::const_view&
            measurements) const = 0;

    /// Function building the measurement ranges buffer
    ///
    /// @param det The detector object
    /// @param n_measurements The number of measurements in the event
    /// @param measurements All measurements in an event
    /// @return The measurement ranges buffer
    ///
    virtual vecmem::data::vector_buffer<
        edm::measurement_collection<default_algebra>::const_view::size_type>
    build_measurement_ranges_buffer(
        const detector_buffer& det,
        const edm::measurement_collection<
            default_algebra>::const_view::size_type n_measurements,
        const edm::measurement_collection<default_algebra>::const_view&
            measurements) const = 0;

    /// Material interaction application kernel launcher
    ///
    /// @param n_threads The number of threads to launch the kernel with
    /// @param config The track finding configuration
    /// @param det The detector object
    /// @param payload The payload for the kernel
    ///
    virtual void apply_interaction_kernel(
        unsigned int n_threads, const finding_config& config,
        const detector_buffer& det,
        const device::apply_interaction_payload& payload) const = 0;

    /// Payload for the @c find_tracks_kernel function
    struct find_tracks_kernel_payload {
        /// The track finding configuration
        const finding_config& config;
        /// The number of input track parameters
        bound_track_parameters_collection_types::const_view::size_type n_params;
        /// The detector object
        const detector_buffer& det;
        /// The measurements view
        const edm::measurement_collection<default_algebra>::const_view&
            measurements;
        /// The input track parameters buffer
        const bound_track_parameters_collection_types::const_view& in_params;
        /// The liveness view of the input track parameters
        const vecmem::data::vector_view<const unsigned int>& in_params_liveness;
        /// The measurement ranges view
        const vecmem::data::vector_view<const unsigned int>& measurement_ranges;
        /// The links buffer
        vecmem::data::vector_view<candidate_link> links;
        /// Index in the link vector at which the previous step starts
        unsigned int prev_links_idx;
        /// Index in the link vector at which the current step starts
        unsigned int curr_links_idx;
        /// The current step index
        unsigned int step;
        /// The output (updated) track parameters buffer
        bound_track_parameters_collection_types::view& out_params;
        /// The liveness view of the output track parameters
        vecmem::data::vector_view<unsigned int>& out_params_liveness;
        /// The tips buffer
        vecmem::data::vector_view<unsigned int>& tips;
        /// The tip lengths buffer
        vecmem::data::vector_view<unsigned int>& tip_lengths;
        /// The number of tracks per seed buffer
        vecmem::data::vector_view<unsigned int>& n_tracks_per_seed;
        /// Temporary track parameters buffer
        bound_track_parameters_collection_types::view& tmp_params;
        /// Temporary links buffer
        vecmem::data::vector_view<candidate_link>& tmp_links;
        /// The Jacobian buffer
        vecmem::data::vector_view<bound_matrix<default_algebra>>& jacobian;
        /// The temporary Jacobian buffer
        vecmem::data::vector_view<bound_matrix<default_algebra>>& tmp_jacobian;
        /// The link predicted parameter buffer
        bound_track_parameters_collection_types::view& link_predicted_parameter;
        /// The link filtered parameter buffer
        bound_track_parameters_collection_types::view& link_filtered_parameter;
    };

    /// Track finding kernel launcher
    ///
    /// @param n_threads The number of threads to launch the kernel with
    /// @param payload The payload for the kernel
    ///
    virtual void find_tracks_kernel(
        unsigned int n_threads,
        const find_tracks_kernel_payload& payload) const = 0;

    /// Launch the kernel sorting the parameters before duplicate removal
    ///
    /// @param n_threads The number of threads to launch the kernel with
    /// @param payload The payload for the kernel
    ///
    virtual void fill_finding_duplicate_removal_sort_keys_kernel(
        unsigned int n_threads,
        const device::fill_finding_duplicate_removal_sort_keys_payload& payload)
        const = 0;

    /// Sort the parameter IDs according to the last measurement index
    ///
    /// @param link_last_measurement The last measurement index per link
    /// @param param_ids The parameter IDs to sort
    ///
    virtual void sort_param_ids_by_last_measurement(
        vecmem::data::vector_view<unsigned int>& link_last_measurement,
        vecmem::data::vector_view<unsigned int>& param_ids) const = 0;

    /// Duplicate removal kernel launcher
    ///
    /// @param n_threads The number of threads to launch the kernel with
    /// @param config The track finding configuration
    /// @param payload The payload for the kernel
    ///
    virtual void remove_duplicates_kernel(
        unsigned int n_threads, const finding_config& config,
        const device::remove_duplicates_payload& payload) const = 0;

    /// Launch the @c fill_finding_propagation_sort_keys kernel
    ///
    /// @param n_threads The number of threads to launch the kernel with
    /// @param payload The payload for the kernel
    ///
    virtual void fill_finding_propagation_sort_keys_kernel(
        unsigned int n_threads,
        const device::fill_finding_propagation_sort_keys_payload& payload)
        const = 0;

    /// Sort the parameter IDs according to a custom set of keys
    ///
    /// @param keys The sort keys
    /// @param param_ids The parameter IDs to sort
    ///
    virtual void sort_param_ids_by_keys(
        vecmem::data::vector_view<device::sort_key>& keys,
        vecmem::data::vector_view<unsigned int>& param_ids) const = 0;

    /// Payload for the @c propagate_to_next_surface_kernel function
    struct propagate_to_next_surface_kernel_payload {
        /// The track finding configuration
        const finding_config& config;
        /// The detector object
        const detector_buffer& det;
        /// The magnetic field object
        const magnetic_field& field;
        /// The vector of track parameters
        bound_track_parameters_collection_types::view& params;
        /// The vector of track parameter liveness values
        vecmem::data::vector_view<unsigned int>& params_liveness;
        /// Sorted parameter identifiers
        const vecmem::data::vector_view<const unsigned int>& param_ids;
        /// The vector of candidate links
        const vecmem::data::vector_view<const candidate_link>& links;
        /// Index in the link vector at which the current step starts
        unsigned int prev_links_idx;
        /// Current CKF step number
        unsigned int step;
        /// The vector of tips
        vecmem::data::vector_view<unsigned int>& tips;
        /// The number of track states per tip
        vecmem::data::vector_view<unsigned int>& tip_lengths;
        /// The temporary Jacobian buffer
        vecmem::data::vector_view<bound_matrix<default_algebra>>& tmp_jacobian;
    };

    /// Launch the @c propagate_to_next_surface kernel
    ///
    /// @param n_threads The number of threads to launch the kernel with
    /// @param payload The payload for the kernel
    ///
    virtual void propagate_to_next_surface_kernel(
        unsigned int n_threads,
        const propagate_to_next_surface_kernel_payload& payload) const = 0;

    /// Launch the @c gather_best_tips_per_measurement kernel
    ///
    /// @param n_threads The number of threads to launch the kernel with
    /// @param payload The payload for the kernel
    ///
    virtual void gather_best_tips_per_measurement_kernel(
        unsigned int n_threads,
        const device::gather_best_tips_per_measurement_payload<default_algebra>&
            payload) const = 0;

    /// Launch the @c gather_measurement_votes kernel
    ///
    /// @param n_threads The number of threads to launch the kernel with
    /// @param payload The payload for the kernel
    ///
    virtual void gather_measurement_votes_kernel(
        unsigned int n_threads,
        const device::gather_measurement_votes_payload& payload) const = 0;

    /// Launch the @c update_tip_length_buffer kernel
    ///
    /// @param n_threads The number of threads to launch the kernel with
    /// @param payload The payload for the kernel
    ///
    virtual void update_tip_length_buffer_kernel(
        unsigned int n_threads,
        const device::update_tip_length_buffer_payload& payload) const = 0;

    /// Launch the @c build_tracks kernel
    ///
    /// @param n_threads The number of threads to launch the kernel with
    /// @param run_mbf_smoother Whether the MBF smoother was run
    /// @param payload The payload for the kernel
    ///
    virtual void build_tracks_kernel(
        unsigned int n_threads, bool run_mbf_smoother,
        const device::build_tracks_payload& payload) const = 0;

    /// @}

    private:
    /// Internal data type
    struct data;
    /// Pointer to internal data
    std::unique_ptr<data> m_data;

};  // class combinatorial_kalman_filter_algorithm

}  // namespace traccc::device
