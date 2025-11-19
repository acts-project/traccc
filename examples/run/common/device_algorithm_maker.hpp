/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/bfield/magnetic_field.hpp"
#include "traccc/clusterization/clustering_config.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/seed_collection.hpp"
#include "traccc/edm/silicon_cell_collection.hpp"
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/edm/track_container.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/finding/finding_config.hpp"
#include "traccc/fitting/fitting_config.hpp"
#include "traccc/geometry/detector_buffer.hpp"
#include "traccc/geometry/silicon_detector_description.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/track_params_estimation_config.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <concepts>
#include <memory>

namespace traccc {

/// Interface for an algorithm maker object
struct device_algorithm_maker {

    /// Virtual destructor
    virtual ~device_algorithm_maker() {}

    /// Access a copy object for the used device
    virtual vecmem::copy& copy() const = 0;

    /// Get the memory resource(s) used by the algorithms
    virtual memory_resource& mr() const = 0;

    /// Wait for the used device to finish all scheduled operations
    virtual void synchronize() const = 0;

    /// Set up the magnetic field for the device
    virtual magnetic_field make_magnetic_field(
        const magnetic_field& bfield, bool texture_memory = false) const = 0;

    /// Construct a clusterization algorithm instance
    virtual std::unique_ptr<
        algorithm<edm::measurement_collection<default_algebra>::buffer(
            const edm::silicon_cell_collection::const_view&,
            const silicon_detector_description::const_view&)>>
    make_clusterization_algorithm(const clustering_config& config) const = 0;

    /// Construct a spacepoint formation algorithm instance
    virtual std::unique_ptr<algorithm<edm::spacepoint_collection::buffer(
        const detector_buffer&,
        const edm::measurement_collection<default_algebra>::const_view&)>>
    make_spacepoint_formation_algorithm() const = 0;

    /// Construct a seeding algorithm instance
    virtual std::unique_ptr<algorithm<edm::seed_collection::buffer(
        const edm::spacepoint_collection::const_view&)>>
    make_seeding_algorithm(const seedfinder_config& finder_config,
                           const spacepoint_grid_config& grid_config,
                           const seedfilter_config& filter_config) const = 0;

    /// Construct a track parameter estimation algorithm instance
    virtual std::unique_ptr<
        algorithm<bound_track_parameters_collection_types::buffer(
            const edm::measurement_collection<default_algebra>::const_view&,
            const edm::spacepoint_collection::const_view&,
            const edm::seed_collection::const_view&, const vector3&)>>
    make_track_params_estimation_algorithm(
        const track_params_estimation_config& config) const = 0;

    /// Construct a track finding algorithm instance
    virtual std::unique_ptr<
        algorithm<edm::track_container<default_algebra>::buffer(
            const detector_buffer&, const magnetic_field&,
            const edm::measurement_collection<default_algebra>::const_view&,
            const bound_track_parameters_collection_types::const_view&)>>
    make_finding_algorithm(const finding_config& config) const = 0;

    /// Construct a track fitting algorithm instance
    virtual std::unique_ptr<
        algorithm<edm::track_container<default_algebra>::buffer(
            const detector_buffer&, const magnetic_field&,
            const edm::track_container<default_algebra>::const_view&)>>
    make_fitting_algorithm(const fitting_config& config) const = 0;

};  // struct algorithm_maker

namespace concepts {

/// Concept specifying a maker of device reconstruction algorithms
template <typename T>
concept device_algorithm_maker =
    std::derived_from<T, traccc::device_algorithm_maker>;

}  // namespace concepts
}  // namespace traccc
