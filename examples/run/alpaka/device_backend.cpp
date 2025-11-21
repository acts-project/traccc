/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "device_backend.hpp"

// Project include(s).
#include "traccc/alpaka/clusterization/clusterization_algorithm.hpp"
#include "traccc/alpaka/finding/combinatorial_kalman_filter_algorithm.hpp"
#include "traccc/alpaka/fitting/kalman_fitting_algorithm.hpp"
#include "traccc/alpaka/seeding/seeding_algorithm.hpp"
#include "traccc/alpaka/seeding/spacepoint_formation_algorithm.hpp"
#include "traccc/alpaka/seeding/track_params_estimation.hpp"
#include "traccc/alpaka/utils/queue.hpp"
#include "traccc/alpaka/utils/vecmem_objects.hpp"

namespace traccc::alpaka {

struct device_backend::impl {

    /// Alpaka queue to use
    queue m_queue;
    /// VecMem objects to use
    vecmem_objects m_vo{m_queue};

    /// Traccc memory resource
    memory_resource m_mr{m_vo.device_mr(), &(m_vo.host_mr())};

};  // struct device_backend::impl

device_backend::device_backend(std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)), m_impl{std::make_unique<impl>()} {}

device_backend::~device_backend() = default;

vecmem::copy& device_backend::copy() const {

    return m_impl->m_vo.async_copy();
}

memory_resource& device_backend::mr() const {

    return m_impl->m_mr;
}

void device_backend::synchronize() const {

    m_impl->m_queue.synchronize();
}

magnetic_field device_backend::make_magnetic_field(const magnetic_field& bfield,
                                                   bool) const {

    return bfield;
}

std::unique_ptr<algorithm<edm::measurement_collection<default_algebra>::buffer(
    const edm::silicon_cell_collection::const_view&,
    const silicon_detector_description::const_view&)>>
device_backend::make_clusterization_algorithm(
    const clustering_config& config) const {

    TRACCC_VERBOSE("Constructing alpaka::clusterization_algorithm");
    return std::make_unique<alpaka::clusterization_algorithm>(
        m_impl->m_mr, m_impl->m_vo.async_copy(), m_impl->m_queue, config,
        logger().clone("alpaka::clusterization_algorithm"));
}

std::unique_ptr<algorithm<edm::spacepoint_collection::buffer(
    const detector_buffer&,
    const edm::measurement_collection<default_algebra>::const_view&)>>
device_backend::make_spacepoint_formation_algorithm() const {

    TRACCC_VERBOSE("Constructing alpaka::spacepoint_formation_algorithm");
    return std::make_unique<alpaka::spacepoint_formation_algorithm>(
        m_impl->m_mr, m_impl->m_vo.async_copy(), m_impl->m_queue,
        logger().clone("alpaka::spacepoint_formation_algorithm"));
}

std::unique_ptr<algorithm<edm::seed_collection::buffer(
    const edm::spacepoint_collection::const_view&)>>
device_backend::make_seeding_algorithm(
    const seedfinder_config& finder_config,
    const spacepoint_grid_config& grid_config,
    const seedfilter_config& filter_config) const {

    TRACCC_VERBOSE("Constructing alpaka::seeding_algorithm");
    return std::make_unique<alpaka::seeding_algorithm>(
        finder_config, grid_config, filter_config, m_impl->m_mr,
        m_impl->m_vo.async_copy(), m_impl->m_queue,
        logger().clone("alpaka::seeding_algorithm"));
}

std::unique_ptr<algorithm<bound_track_parameters_collection_types::buffer(
    const edm::measurement_collection<default_algebra>::const_view&,
    const edm::spacepoint_collection::const_view&,
    const edm::seed_collection::const_view&, const vector3&)>>
device_backend::make_track_params_estimation_algorithm(
    const track_params_estimation_config& config) const {

    TRACCC_VERBOSE("Constructing alpaka::track_params_estimation");
    return std::make_unique<alpaka::track_params_estimation>(
        config, m_impl->m_mr, m_impl->m_vo.async_copy(), m_impl->m_queue,
        logger().clone("alpaka::track_params_estimation"));
}

std::unique_ptr<algorithm<edm::track_container<default_algebra>::buffer(
    const detector_buffer&, const magnetic_field&,
    const edm::measurement_collection<default_algebra>::const_view&,
    const bound_track_parameters_collection_types::const_view&)>>
device_backend::make_finding_algorithm(const finding_config& config) const {

    TRACCC_VERBOSE(
        "Constructing alpaka::combinatorial_kalman_filter_algorithm");
    return std::make_unique<alpaka::combinatorial_kalman_filter_algorithm>(
        config, m_impl->m_mr, m_impl->m_vo.async_copy(), m_impl->m_queue,
        logger().clone("alpaka::combinatorial_kalman_filter_algorithm"));
}

std::unique_ptr<algorithm<edm::track_container<default_algebra>::buffer(
    const detector_buffer&, const magnetic_field&,
    const edm::track_container<default_algebra>::const_view&)>>
device_backend::make_fitting_algorithm(const fitting_config& config) const {

    TRACCC_VERBOSE("Constructing alpaka::kalman_fitting_algorithm");
    return std::make_unique<alpaka::kalman_fitting_algorithm>(
        config, m_impl->m_mr, m_impl->m_vo.async_copy(), m_impl->m_queue,
        logger().clone("alpaka::kalman_fitting_algorithm"));
}

}  // namespace traccc::alpaka
