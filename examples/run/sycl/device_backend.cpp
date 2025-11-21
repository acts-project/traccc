/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "device_backend.hpp"

// Project include(s).
#include "traccc/sycl/clusterization/clusterization_algorithm.hpp"
#include "traccc/sycl/clusterization/measurement_sorting_algorithm.hpp"
#include "traccc/sycl/finding/combinatorial_kalman_filter_algorithm.hpp"
#include "traccc/sycl/fitting/kalman_fitting_algorithm.hpp"
#include "traccc/sycl/seeding/seeding_algorithm.hpp"
#include "traccc/sycl/seeding/silicon_pixel_spacepoint_formation_algorithm.hpp"
#include "traccc/sycl/seeding/track_params_estimation.hpp"
#include "traccc/sycl/utils/make_magnetic_field.hpp"
#include "traccc/sycl/utils/queue_wrapper.hpp"

// VecMem include(s).
#include <vecmem/memory/sycl/device_memory_resource.hpp>
#include <vecmem/memory/sycl/host_memory_resource.hpp>
#include <vecmem/utils/sycl/async_copy.hpp>
#include <vecmem/utils/sycl/queue_wrapper.hpp>

namespace traccc::sycl {

struct device_backend::impl {

    /// (VecMem) SYCL queue to use
    vecmem::sycl::queue_wrapper m_vecmem_queue;
    /// (Traccc) SYCL queue wrapper
    traccc::sycl::queue_wrapper m_traccc_queue{m_vecmem_queue.queue()};

    /// Host memory resource
    vecmem::sycl::host_memory_resource m_host_mr{m_vecmem_queue};
    /// Device memory resource
    vecmem::sycl::device_memory_resource m_device_mr{m_vecmem_queue};
    /// Traccc memory resource
    memory_resource m_mr{m_device_mr, &m_host_mr};

    /// (Asynchronous) Memory copy object
    vecmem::sycl::async_copy m_copy{m_vecmem_queue};

};  // struct device_backend::impl

device_backend::device_backend(std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)), m_impl{std::make_unique<impl>()} {}

device_backend::~device_backend() = default;

vecmem::copy& device_backend::copy() const {

    return m_impl->m_copy;
}

memory_resource& device_backend::mr() const {

    return m_impl->m_mr;
}

void device_backend::synchronize() const {

    m_impl->m_vecmem_queue.synchronize();
}

magnetic_field device_backend::make_magnetic_field(const magnetic_field& bfield,
                                                   bool) const {

    return sycl::make_magnetic_field(bfield, m_impl->m_traccc_queue);
}

std::unique_ptr<algorithm<edm::measurement_collection<default_algebra>::buffer(
    const edm::silicon_cell_collection::const_view&,
    const silicon_detector_description::const_view&)>>
device_backend::make_clusterization_algorithm(
    const clustering_config& config) const {

    TRACCC_VERBOSE("Constructing sycl::clusterization_algorithm");
    return std::make_unique<sycl::clusterization_algorithm>(
        m_impl->m_mr, m_impl->m_copy, m_impl->m_traccc_queue, config,
        logger().clone("sycl::clusterization_algorithm"));
}

std::unique_ptr<algorithm<edm::measurement_collection<default_algebra>::buffer(
    const edm::measurement_collection<default_algebra>::const_view&)>>
device_backend::make_measurement_sorting_algorithm() const {

    TRACCC_VERBOSE("Constructing sycl::measurement_sorting_algorithm");
    return std::make_unique<sycl::measurement_sorting_algorithm>(
        m_impl->m_mr, m_impl->m_copy, m_impl->m_traccc_queue,
        logger().clone("sycl::measurement_sorting_algorithm"));
}

std::unique_ptr<algorithm<edm::spacepoint_collection::buffer(
    const detector_buffer&,
    const edm::measurement_collection<default_algebra>::const_view&)>>
device_backend::make_spacepoint_formation_algorithm() const {

    TRACCC_VERBOSE(
        "Constructing sycl::silicon_pixel_spacepoint_formation_algorithm");
    return std::make_unique<sycl::silicon_pixel_spacepoint_formation_algorithm>(
        m_impl->m_mr, m_impl->m_copy, m_impl->m_traccc_queue,
        logger().clone("sycl::silicon_pixel_spacepoint_formation_algorithm"));
}

std::unique_ptr<algorithm<edm::seed_collection::buffer(
    const edm::spacepoint_collection::const_view&)>>
device_backend::make_seeding_algorithm(
    const seedfinder_config& finder_config,
    const spacepoint_grid_config& grid_config,
    const seedfilter_config& filter_config) const {

    TRACCC_VERBOSE("Constructing sycl::seeding_algorithm");
    return std::make_unique<sycl::seeding_algorithm>(
        finder_config, grid_config, filter_config, m_impl->m_mr, m_impl->m_copy,
        m_impl->m_traccc_queue, logger().clone("sycl::seeding_algorithm"));
}

std::unique_ptr<algorithm<bound_track_parameters_collection_types::buffer(
    const edm::measurement_collection<default_algebra>::const_view&,
    const edm::spacepoint_collection::const_view&,
    const edm::seed_collection::const_view&, const vector3&)>>
device_backend::make_track_params_estimation_algorithm(
    const track_params_estimation_config& config) const {

    TRACCC_VERBOSE("Constructing sycl::track_params_estimation");
    return std::make_unique<sycl::track_params_estimation>(
        config, m_impl->m_mr, m_impl->m_copy, m_impl->m_traccc_queue,
        logger().clone("sycl::track_params_estimation"));
}

std::unique_ptr<algorithm<edm::track_container<default_algebra>::buffer(
    const detector_buffer&, const magnetic_field&,
    const edm::measurement_collection<default_algebra>::const_view&,
    const bound_track_parameters_collection_types::const_view&)>>
device_backend::make_finding_algorithm(const finding_config& config) const {

    TRACCC_VERBOSE("Constructing sycl::combinatorial_kalman_filter_algorithm");
    return std::make_unique<sycl::combinatorial_kalman_filter_algorithm>(
        config, m_impl->m_mr, m_impl->m_copy, m_impl->m_traccc_queue,
        logger().clone("sycl::combinatorial_kalman_filter_algorithm"));
}

std::unique_ptr<algorithm<edm::track_container<default_algebra>::buffer(
    const edm::track_container<default_algebra>::const_view&)>>
device_backend::make_ambiguity_resolution_algorithm(
    const ambiguity_resolution_config&) const {

    TRACCC_DEBUG(
        "No ambiguity resolution algorithm implemented for the SYCL backend");
    return {};
}

std::unique_ptr<algorithm<edm::track_container<default_algebra>::buffer(
    const detector_buffer&, const magnetic_field&,
    const edm::track_container<default_algebra>::const_view&)>>
device_backend::make_fitting_algorithm(const fitting_config& config) const {

    TRACCC_VERBOSE("Constructing sycl::kalman_fitting_algorithm");
    return std::make_unique<sycl::kalman_fitting_algorithm>(
        config, m_impl->m_mr, m_impl->m_copy, m_impl->m_traccc_queue,
        logger().clone("sycl::kalman_fitting_algorithm"));
}

}  // namespace traccc::sycl
