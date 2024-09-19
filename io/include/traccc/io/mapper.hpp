/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/particle.hpp"
#include "traccc/edm/silicon_cell_collection.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/geometry/silicon_detector_description.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

// System include(s).
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>

namespace traccc {

using hit_id = uint64_t;
using particle_id = uint64_t;
using particle_map = std::map<particle_id, particle>;
using hit_particle_map = std::map<spacepoint, particle>;
using hit_map = std::map<hit_id, spacepoint>;
using hit_cell_map = std::map<spacepoint, edm::silicon_cell_collection::host>;
using geoId_link_map = std::map<geometry_id, unsigned int>;
using particle_cell_map =
    std::map<particle, edm::silicon_cell_collection::host>;
using measurement_cell_map =
    std::map<measurement, edm::silicon_cell_collection::host>;
using measurement_particle_map =
    std::map<measurement, std::map<particle, uint64_t>>;

particle_map generate_particle_map(std::size_t event,
                                   const std::string& particle_dir);

hit_particle_map generate_hit_particle_map(
    std::size_t event, const std::string& hits_dir,
    const std::string& particle_dir,
    const geoId_link_map& link_map = geoId_link_map());

hit_map generate_hit_map(std::size_t event, const std::string& hits_dir);

hit_cell_map generate_hit_cell_map(
    std::size_t event, const std::string& cells_dir,
    const std::string& hits_dir, vecmem::memory_resource& resource,
    const geoId_link_map& link_map = geoId_link_map());

particle_cell_map generate_particle_cell_map(
    std::size_t event, const std::string& cells_dir,
    const std::string& hits_dir, const std::string& particle_dir,
    vecmem::memory_resource& resource,
    const geoId_link_map& link_map = geoId_link_map());

measurement_cell_map generate_measurement_cell_map(
    std::size_t event, const std::string& cells_dir,
    const silicon_detector_description::host& dd,
    vecmem::memory_resource& resource);

measurement_particle_map generate_measurement_particle_map(
    std::size_t event, const std::string& cells_dir,
    const std::string& hits_dir, const std::string& particle_dir,
    const silicon_detector_description::host& dd,
    vecmem::memory_resource& resource);

measurement_particle_map generate_measurement_particle_map(
    std::size_t event, const std::string& hits_dir,
    const std::string& particle_dir,
    const silicon_detector_description::host& dd,
    vecmem::memory_resource& resource);

}  // namespace traccc
