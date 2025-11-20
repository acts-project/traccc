/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "../common/device_backend.hpp"
#include "traccc/utils/messaging.hpp"

// System include(s).
#include <memory>

namespace traccc::cuda {

/// CUDA Device Backend
class device_backend : public traccc::device_backend, public messaging {

    public:
    /// Constructor
    ///
    /// @param logger The logger to use
    ///
    device_backend(
        std::unique_ptr<const Logger> logger = getDummyLogger().clone());
    /// Destructor
    ~device_backend();

    /// @name Function(s) implemented from @c traccc::device_backend
    /// @{

    /// Access a copy object for the used device
    vecmem::copy& copy() const override;

    /// Get the memory resource(s) used by the algorithms
    memory_resource& mr() const override;

    /// Wait for the used device to finish all scheduled operations
    void synchronize() const override;

    /// Set up the magnetic field for the device
    magnetic_field make_magnetic_field(
        const magnetic_field& bfield,
        bool texture_memory = false) const override;

    /// Construct a clusterization algorithm instance
    std::unique_ptr<
        algorithm<edm::measurement_collection<default_algebra>::buffer(
            const edm::silicon_cell_collection::const_view&,
            const silicon_detector_description::const_view&)>>
    make_clusterization_algorithm(
        const clustering_config& config) const override;

    /// Construct a spacepoint formation algorithm instance
    std::unique_ptr<algorithm<edm::spacepoint_collection::buffer(
        const detector_buffer&,
        const edm::measurement_collection<default_algebra>::const_view&)>>
    make_spacepoint_formation_algorithm() const override;

    /// Construct a seeding algorithm instance
    std::unique_ptr<algorithm<edm::seed_collection::buffer(
        const edm::spacepoint_collection::const_view&)>>
    make_seeding_algorithm(
        const seedfinder_config& finder_config,
        const spacepoint_grid_config& grid_config,
        const seedfilter_config& filter_config) const override;

    /// Construct a track parameter estimation algorithm instance
    std::unique_ptr<algorithm<bound_track_parameters_collection_types::buffer(
        const edm::measurement_collection<default_algebra>::const_view&,
        const edm::spacepoint_collection::const_view&,
        const edm::seed_collection::const_view&, const vector3&)>>
    make_track_params_estimation_algorithm(
        const track_params_estimation_config& config) const override;

    /// Construct a track finding algorithm instance
    std::unique_ptr<algorithm<edm::track_container<default_algebra>::buffer(
        const detector_buffer&, const magnetic_field&,
        const edm::measurement_collection<default_algebra>::const_view&,
        const bound_track_parameters_collection_types::const_view&)>>
    make_finding_algorithm(const finding_config& config) const override;

    /// Construct a track fitting algorithm instance
    std::unique_ptr<algorithm<edm::track_container<default_algebra>::buffer(
        const detector_buffer&, const magnetic_field&,
        const edm::track_container<default_algebra>::const_view&)>>
    make_fitting_algorithm(const fitting_config& config) const override;

    /// @}

    private:
    /// Implementation class
    struct impl;
    /// PIMPL data object
    std::unique_ptr<impl> m_impl;

};  // class device_backend

}  // namespace traccc::cuda
