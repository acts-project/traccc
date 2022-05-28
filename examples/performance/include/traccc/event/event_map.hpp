/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/io/mapper.hpp"

namespace traccc {

struct event_map {

    event_map() = delete;

    event_map(std::size_t event, const std::string& detector_file,
              const std::string& digi_config_file, const std::string& cell_dir,
              const std::string& hit_dir, const std::string particle_dir,
              vecmem::memory_resource& resource) {

        ptc_map = generate_particle_map(event, particle_dir);
        meas_ptc_map = generate_measurement_particle_map(
            event, detector_file, digi_config_file, cell_dir, hit_dir,
            particle_dir, resource);
    }

    event_map(std::size_t event, const std::string& detector_file,
              const std::string& hit_dir, const std::string particle_dir,
              vecmem::memory_resource& resource) {

        ptc_map = generate_particle_map(event, particle_dir);
        meas_ptc_map = generate_measurement_particle_map(
            event, detector_file, hit_dir, particle_dir, resource);
    }

    particle_map ptc_map;
    measurement_particle_map meas_ptc_map;
};

}  // namespace traccc