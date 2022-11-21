/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/io/mapper.hpp"

#include "csv/make_cell_reader.hpp"

namespace traccc {

hit_cell_map generate_hit_cell_map(size_t event, const std::string& cells_dir,
                                   const std::string& hits_dir) {

    hit_cell_map result;

    auto hmap = generate_hit_map(event, hits_dir);

    // Read the cells from the relevant event file
    std::string io_cells_file =
        data_directory() + cells_dir + get_event_filename(event, "-cells.csv");

    auto creader = io::csv::make_cell_reader(io_cells_file);

    io::csv::cell iocell;

    while (creader.read(iocell)) {
        result[hmap[iocell.hit_id]].push_back(cell{
            iocell.channel0, iocell.channel1, iocell.value, iocell.timestamp});
    }

    return result;
}

}  // namespace traccc
