/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/io/csv/make_particle_reader.hpp"

#include "traccc/io/csv/particle.hpp"

namespace traccc::io::csv {

dfe::NamedTupleCsvReader<particle> make_particle_reader(
    std::string_view filename) {

    return {filename.data(),
            {"particle_id", "particle_type", "process", "vx", "vy", "vz", "vt",
             "px", "py", "pz", "m", "q"}};
}

dfe::NamedTupleCsvReader<particle_with_split_id>
make_particle_reader_with_split_id(std::string_view filename) {

    return {filename.data(),
            {"particle_id_pv", "particle_id_sv", "particle_id_part",
             "particle_id_gen", "particle_id_subpart", "particle_type",
             "process", "vx", "vy", "vz", "vt", "px", "py", "pz", "m", "q"}};
}

}  // namespace traccc::io::csv
