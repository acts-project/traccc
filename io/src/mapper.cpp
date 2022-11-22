/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/io/mapper.hpp"

#include "csv/make_cell_reader.hpp"
#include "csv/make_hit_reader.hpp"

namespace traccc {

hit_particle_map generate_hit_particle_map(size_t event,
                                           const std::string& hits_dir,
                                           const std::string& particle_dir) {
    hit_particle_map result;

    auto pmap = generate_particle_map(event, particle_dir);

    // Read the hits from the relevant event file
    std::string io_hits_file = io::data_directory() + hits_dir +
                               io::get_event_filename(event, "-hits.csv");

    auto hreader = io::csv::make_hit_reader(io_hits_file);

    io::csv::hit iohit;

    while (hreader.read(iohit)) {

        spacepoint sp;
        sp.global = {iohit.tx, iohit.ty, iohit.tz};

        particle ptc = pmap[iohit.particle_id];

        result[sp] = ptc;
    }

    return result;
}

hit_map generate_hit_map(size_t event, const std::string& hits_dir) {
    hit_map result;

    // Read the hits from the relevant event file
    std::string io_hits_file = io::data_directory() + hits_dir +
                               io::get_event_filename(event, "-hits.csv");

    auto hreader = io::csv::make_hit_reader(io_hits_file);

    io::csv::hit iohit;

    // Read the hits from the relevant event file
    std::string io_meas_hit_id_file =
        io::data_directory() + hits_dir +
        io::get_event_filename(event, "-measurement-simhit-map.csv");

    meas_hit_id_reader mhid_reader(io_meas_hit_id_file,
                                   {"measurement_id", "hit_id"});

    csv_meas_hit_id mh_id;

    std::map<uint64_t, uint64_t> mh_id_map;

    while (mhid_reader.read(mh_id)) {
        mh_id_map[mh_id.hit_id] = mh_id.measurement_id;
    }

    hit_id hid = 0;
    while (hreader.read(iohit)) {

        spacepoint sp;
        sp.global = {iohit.tx, iohit.ty, iohit.tz};

        // result[hid] = sp;
        result[mh_id_map[hid]] = sp;

        hid++;
    }

    return result;
}

hit_cell_map generate_hit_cell_map(size_t event, const std::string& cells_dir,
                                   const std::string& hits_dir) {

    hit_cell_map result;

    auto hmap = generate_hit_map(event, hits_dir);

    // Read the cells from the relevant event file
    std::string io_cells_file = io::data_directory() + cells_dir +
                                io::get_event_filename(event, "-cells.csv");

    auto creader = io::csv::make_cell_reader(io_cells_file);

    io::csv::cell iocell;

    while (creader.read(iocell)) {
        result[hmap[iocell.hit_id]].push_back(cell{
            iocell.channel0, iocell.channel1, iocell.value, iocell.timestamp});
    }

    return result;
}

}  // namespace traccc
