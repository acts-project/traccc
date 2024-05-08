/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "read_particles.hpp"

#include "traccc/io/csv/make_particle_reader.hpp"

namespace traccc::io::csv {

void read_particles(particle_collection_types::host& particles,
                    std::string_view filename) {

    // Construct the particle reader object.
    auto reader = make_particle_reader(filename);

    // Read the particles from the input file.
    csv::particle part;
    while (reader.read(part)) {
        particles.push_back({part.particle_id, part.particle_type, part.process,
                             point3{part.vx, part.vy, part.vz}, part.vt,
                             vector3{part.px, part.py, part.pz}, part.m,
                             part.q});
    }
}

}  // namespace traccc::io::csv
