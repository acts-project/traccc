/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/io/csv/make_hit_reader.hpp"

#include "traccc/io/csv/hit.hpp"

namespace traccc::io::csv {

dfe::NamedTupleCsvReader<hit> make_hit_reader(std::string_view filename) {

    return {filename.data(),
            {"particle_id", "geometry_id", "tx", "ty", "tz", "tt", "tpx", "tpy",
             "tpz", "te", "deltapx", "deltapy", "deltapz", "deltae", "index"}};
}

dfe::NamedTupleCsvReader<hit_with_split_particle_id>
make_hit_reader_with_split_particle_id(std::string_view filename) {

    return {filename.data(),
            {"particle_id_pv", "particle_id_sv", "particle_id_part",
             "particle_id_gen", "particle_id_subpart", "geometry_id", "tx",
             "ty", "tz", "tt", "tpx", "tpy", "tpz", "te", "deltapx", "deltapy",
             "deltapz", "deltae", "index"}};
}

}  // namespace traccc::io::csv
