/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/sycl/utils/algorithm_base.hpp"

// Project include(s).
#include "traccc/finding/device/combinatorial_kalman_filter_algorithm.hpp"

// System include(s).
#include <memory>

namespace traccc::sycl {

/// CKF track finding algorithm using SYCL
class combinatorial_kalman_filter_algorithm
    : public device::combinatorial_kalman_filter_algorithm,
      public sycl::algorithm_base {

    public:
    /// Constructor with the algorithm's configuration
    combinatorial_kalman_filter_algorithm(
        const finding_config& config, const traccc::memory_resource& mr,
        vecmem::copy& copy, queue_wrapper& queue,
        std::unique_ptr<const Logger> logger = getDummyLogger().clone());

    private:
    /// @name Function(s) inherited from
    ///       @c traccc::device::combinatorial_kalman_filter_algorithm
    /// @{

    /// Function meant to perform sanity checks on the input data
    ///
    /// @param measurements All measurements in an event
    /// @return @c true if the input data is valid, @c false otherwise
    ///
    bool input_is_valid(
        const edm::measurement_collection<default_algebra>::const_view&
            measurements) const override;

    /// Function building the measurement ranges buffer
    ///
    /// @param det The detector object
    /// @param n_measurements The number of measurements in the event
    /// @param measurements All measurements in an event
    /// @return The measurement ranges buffer
    ///
    vecmem::data::vector_buffer<
        edm::measurement_collection<default_algebra>::const_view::size_type>
    build_measurement_ranges_buffer(
        const detector_buffer& det,
        const edm::measurement_collection<
            default_algebra>::const_view::size_type n_measurements,
        const edm::measurement_collection<default_algebra>::const_view&
            measurements) const override;

    /// Material interaction application kernel launcher
    ///
    /// @param n_threads The number of threads to launch the kernel with
    /// @param payload The payload for the kernel
    ///
    void apply_interaction_kernel(
        unsigned int n_threads,
        const apply_interaction_kernel_payload& payload) const override;

    /// Track finding kernel launcher
    ///
    /// @param n_threads The number of threads to launch the kernel with
    /// @param payload The payload for the kernel
    ///
    void find_tracks_kernel(
        unsigned int n_threads,
        const find_tracks_kernel_payload& payload) const override;

    /// Launch the kernel sorting the parameters before duplicate removal
    ///
    /// @param n_threads The number of threads to launch the kernel with
    /// @param payload The payload for the kernel
    ///
    void fill_finding_duplicate_removal_sort_keys_kernel(
        unsigned int n_threads,
        const device::fill_finding_duplicate_removal_sort_keys_payload& payload)
        const override;

    /// Sort the parameter IDs according to the last measurement index
    ///
    /// @param link_last_measurement The last measurement index per link
    /// @param param_ids The parameter IDs to sort
    ///
    void sort_param_ids_by_last_measurement(
        vecmem::data::vector_view<unsigned int>& link_last_measurement,
        vecmem::data::vector_view<unsigned int>& param_ids) const override;

    /// Duplicate removal kernel launcher
    ///
    /// @param n_threads The number of threads to launch the kernel with
    /// @param config The track finding configuration
    /// @param payload The payload for the kernel
    ///
    void remove_duplicates_kernel(
        unsigned int n_threads, const finding_config& config,
        const device::remove_duplicates_payload& payload) const override;

    /// Launch the @c fill_finding_propagation_sort_keys kernel
    ///
    /// @param n_threads The number of threads to launch the kernel with
    /// @param payload The payload for the kernel
    ///
    void fill_finding_propagation_sort_keys_kernel(
        unsigned int n_threads,
        const device::fill_finding_propagation_sort_keys_payload& payload)
        const override;

    /// Sort the parameter IDs according to a custom set of keys
    ///
    /// @param keys The sort keys
    /// @param param_ids The parameter IDs to sort
    ///
    void sort_param_ids_by_keys(
        vecmem::data::vector_view<device::sort_key>& keys,
        vecmem::data::vector_view<unsigned int>& param_ids) const override;

    /// Launch the @c propagate_to_next_surface kernel
    ///
    /// @param n_threads The number of threads to launch the kernel with
    /// @param payload The payload for the kernel
    ///
    void propagate_to_next_surface_kernel(
        unsigned int n_threads,
        const propagate_to_next_surface_kernel_payload& payload) const override;

    /// Launch the @c gather_best_tips_per_measurement kernel
    ///
    /// @param n_threads The number of threads to launch the kernel with
    /// @param payload The payload for the kernel
    ///
    void gather_best_tips_per_measurement_kernel(
        unsigned int n_threads,
        const device::gather_best_tips_per_measurement_payload<default_algebra>&
            payload) const override;

    /// Launch the @c gather_measurement_votes kernel
    ///
    /// @param n_threads The number of threads to launch the kernel with
    /// @param payload The payload for the kernel
    ///
    void gather_measurement_votes_kernel(
        unsigned int n_threads,
        const device::gather_measurement_votes_payload& payload) const override;

    /// Launch the @c update_tip_length_buffer kernel
    ///
    /// @param n_threads The number of threads to launch the kernel with
    /// @param payload The payload for the kernel
    ///
    void update_tip_length_buffer_kernel(
        unsigned int n_threads,
        const device::update_tip_length_buffer_payload& payload) const override;

    /// Launch the @c build_tracks kernel
    ///
    /// @param n_threads The number of threads to launch the kernel with
    /// @param run_mbf_smoother Whether the MBF smoother was run
    /// @param payload The payload for the kernel
    ///
    void build_tracks_kernel(
        unsigned int n_threads, bool run_mbf_smoother,
        const device::build_tracks_payload& payload) const override;

    /// @}

};  // class combinatorial_kalman_filter_algorithm

}  // namespace traccc::sycl
