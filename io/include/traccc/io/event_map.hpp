/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/io/mapper.hpp"

// Project include(s).
#include "traccc/geometry/silicon_detector_description.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

// System include(s).
#include <string>

namespace traccc {

struct event_map {

    event_map() = delete;

    event_map(std::size_t event, const std::string& cell_dir,
              const std::string& hit_dir, const std::string particle_dir,
              const silicon_detector_description::host& dd,
              vecmem::memory_resource& resource) {

        ptc_map = generate_particle_map(event, particle_dir);
        meas_ptc_map = generate_measurement_particle_map(
            event, cell_dir, hit_dir, particle_dir, dd, resource);
    }

    event_map(std::size_t event, const std::string& hit_dir,
              const std::string particle_dir,
              const silicon_detector_description::host& dd,
              vecmem::memory_resource& resource) {

        ptc_map = generate_particle_map(event, particle_dir);
        meas_ptc_map = generate_measurement_particle_map(
            event, hit_dir, particle_dir, dd, resource);
    }

    particle_map ptc_map;
    measurement_particle_map meas_ptc_map;
};

}  // namespace traccc