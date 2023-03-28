/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/alt_measurement.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/particle.hpp"
#include "traccc/edm/spacepoint.hpp"

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
using hit_cell_map = std::map<spacepoint, std::vector<cell>>;
using geoId_link_map = std::map<geometry_id, unsigned int>;
using cell_particle_map = std::map<cell, particle>;
using measurement_cell_map = std::map<alt_measurement, vecmem::vector<cell>>;
using measurement_particle_map =
    std::map<alt_measurement, std::map<particle, uint64_t>>;

particle_map generate_particle_map(std::size_t event,
                                   const std::string& particle_dir);

hit_particle_map generate_hit_particle_map(
    std::size_t event, const std::string& hits_dir,
    const std::string& particle_dir,
    const geoId_link_map& link_map = geoId_link_map());

hit_map generate_hit_map(std::size_t event, const std::string& hits_dir);

hit_cell_map generate_hit_cell_map(
    std::size_t event, const std::string& cells_dir,
    const std::string& hits_dir,
    const geoId_link_map& link_map = geoId_link_map());

cell_particle_map generate_cell_particle_map(
    std::size_t event, const std::string& cells_dir,
    const std::string& hits_dir, const std::string& particle_dir,
    const geoId_link_map& link_map = geoId_link_map());

std::tuple<measurement_cell_map, cell_module_collection_types::host>
generate_measurement_cell_map(std::size_t event,
                              const std::string& detector_file,
                              const std::string& digi_config_file,
                              const std::string& cells_dir,
                              vecmem::memory_resource& resource);

measurement_particle_map generate_measurement_particle_map(
    std::size_t event, const std::string& detector_file,
    const std::string& digi_config_file, const std::string& cells_dir,
    const std::string& hits_dir, const std::string& particle_dir,
    vecmem::memory_resource& resource);

measurement_particle_map generate_measurement_particle_map(
    std::size_t event, const std::string& detector_file,
    const std::string& hits_dir, const std::string& particle_dir,
    vecmem::memory_resource& resource);

}  // namespace traccc
