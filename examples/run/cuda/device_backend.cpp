/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "device_backend.hpp"

// Project include(s).
#include "traccc/cuda/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"
#include "traccc/cuda/clusterization/clusterization_algorithm.hpp"
#include "traccc/cuda/clusterization/measurement_sorting_algorithm.hpp"
#include "traccc/cuda/finding/combinatorial_kalman_filter_algorithm.hpp"
#include "traccc/cuda/fitting/kalman_fitting_algorithm.hpp"
#include "traccc/cuda/seeding/seeding_algorithm.hpp"
#include "traccc/cuda/seeding/spacepoint_formation_algorithm.hpp"
#include "traccc/cuda/seeding/track_params_estimation.hpp"
#include "traccc/cuda/utils/make_magnetic_field.hpp"
#include "traccc/cuda/utils/stream.hpp"

// VecMem include(s).
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>

namespace traccc::cuda {

struct device_backend::impl {

    /// CUDA stream to use
    stream m_stream;

    /// Host memory resource
    vecmem::cuda::host_memory_resource m_host_mr;
    /// Device memory resource
    vecmem::cuda::device_memory_resource m_device_mr{m_stream.device()};
    /// Traccc memory resource
    memory_resource m_mr{m_device_mr, &m_host_mr};

    /// (Asynchronous) Memory copy object
    vecmem::cuda::async_copy m_copy{m_stream.cudaStream()};

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

    m_impl->m_stream.synchronize();
}

magnetic_field device_backend::make_magnetic_field(
    const magnetic_field& bfield, const bool texture_memory) const {

    return cuda::make_magnetic_field(
        bfield, (texture_memory ? magnetic_field_storage::texture_memory
                                : magnetic_field_storage::global_memory));
}

std::unique_ptr<algorithm<edm::measurement_collection<default_algebra>::buffer(
    const edm::silicon_cell_collection::const_view&,
    const silicon_detector_description::const_view&)>>
device_backend::make_clusterization_algorithm(
    const clustering_config& config) const {

    TRACCC_VERBOSE("Constructing cuda::clusterization_algorithm");
    return std::make_unique<cuda::clusterization_algorithm>(
        m_impl->m_mr, m_impl->m_copy, m_impl->m_stream, config,
        logger().clone("cuda::clusterization_algorithm"));
}

std::unique_ptr<algorithm<edm::measurement_collection<default_algebra>::buffer(
    const edm::measurement_collection<default_algebra>::const_view&)>>
device_backend::make_measurement_sorting_algorithm() const {

    TRACCC_VERBOSE("Constructing cuda::measurement_sorting_algorithm");
    return std::make_unique<cuda::measurement_sorting_algorithm>(
        m_impl->m_mr, m_impl->m_copy, m_impl->m_stream,
        logger().clone("cuda::measurement_sorting_algorithm"));
}

std::unique_ptr<algorithm<edm::spacepoint_collection::buffer(
    const detector_buffer&,
    const edm::measurement_collection<default_algebra>::const_view&)>>
device_backend::make_spacepoint_formation_algorithm() const {

    TRACCC_VERBOSE("Constructing cuda::spacepoint_formation_algorithm");
    return std::make_unique<cuda::spacepoint_formation_algorithm>(
        m_impl->m_mr, m_impl->m_copy, m_impl->m_stream,
        logger().clone("cuda::spacepoint_formation_algorithm"));
}

std::unique_ptr<algorithm<edm::seed_collection::buffer(
    const edm::spacepoint_collection::const_view&)>>
device_backend::make_seeding_algorithm(
    const seedfinder_config& finder_config,
    const spacepoint_grid_config& grid_config,
    const seedfilter_config& filter_config) const {

    TRACCC_VERBOSE("Constructing cuda::seeding_algorithm");
    return std::make_unique<cuda::seeding_algorithm>(
        finder_config, grid_config, filter_config, m_impl->m_mr, m_impl->m_copy,
        m_impl->m_stream, logger().clone("cuda::seeding_algorithm"));
}

std::unique_ptr<algorithm<bound_track_parameters_collection_types::buffer(
    const edm::measurement_collection<default_algebra>::const_view&,
    const edm::spacepoint_collection::const_view&,
    const edm::seed_collection::const_view&, const vector3&)>>
device_backend::make_track_params_estimation_algorithm(
    const track_params_estimation_config& config) const {

    TRACCC_VERBOSE("Constructing cuda::track_params_estimation");
    return std::make_unique<cuda::track_params_estimation>(
        config, m_impl->m_mr, m_impl->m_copy, m_impl->m_stream,
        logger().clone("cuda::track_params_estimation"));
}

std::unique_ptr<algorithm<edm::track_container<default_algebra>::buffer(
    const detector_buffer&, const magnetic_field&,
    const edm::measurement_collection<default_algebra>::const_view&,
    const bound_track_parameters_collection_types::const_view&)>>
device_backend::make_finding_algorithm(const finding_config& config) const {

    TRACCC_VERBOSE("Constructing cuda::combinatorial_kalman_filter_algorithm");
    return std::make_unique<cuda::combinatorial_kalman_filter_algorithm>(
        config, m_impl->m_mr, m_impl->m_copy, m_impl->m_stream,
        logger().clone("cuda::combinatorial_kalman_filter_algorithm"));
}

std::unique_ptr<algorithm<edm::track_container<default_algebra>::buffer(
    const edm::track_container<default_algebra>::const_view&)>>
device_backend::make_ambiguity_resolution_algorithm(
    const ambiguity_resolution_config& config) const {

    TRACCC_VERBOSE("Constructing cuda::greedy_ambiguity_resolution_algorithm");
    return std::make_unique<cuda::greedy_ambiguity_resolution_algorithm>(
        config, m_impl->m_mr, m_impl->m_copy, m_impl->m_stream,
        logger().clone("cuda::greedy_ambiguity_resolution_algorithm"));
}

std::unique_ptr<algorithm<edm::track_container<default_algebra>::buffer(
    const detector_buffer&, const magnetic_field&,
    const edm::track_container<default_algebra>::const_view&)>>
device_backend::make_fitting_algorithm(const fitting_config& config) const {

    TRACCC_VERBOSE("Constructing cuda::kalman_fitting_algorithm");
    return std::make_unique<cuda::kalman_fitting_algorithm>(
        config, m_impl->m_mr, m_impl->m_copy, m_impl->m_stream,
        logger().clone("cuda::kalman_fitting_algorithm"));
}

}  // namespace traccc::cuda
