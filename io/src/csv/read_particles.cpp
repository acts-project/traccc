/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "read_particles.hpp"

#include "make_particle_reader.hpp"

namespace traccc::io::csv {

particle_collection_types::host read_particles(std::string_view filename,
                                               vecmem::memory_resource* mr) {

    // Construct the particle reader object.
    auto reader = make_particle_reader(filename);

    // Create the result collection.
    particle_collection_types::host result;
    if (mr != nullptr) {
        result = particle_collection_types::host{mr};
    }

    // Read the particles from the input file.
    csv::particle part;
    while (reader.read(part)) {
        result.push_back({part.particle_id, part.particle_type, part.process,
                          point3{part.vx, part.vy, part.vz}, part.vt,
                          vector3{part.px, part.py, part.pz}, part.m, part.q});
    }

    // Return the collection.
    return result;
}

}  // namespace traccc::io::csv
