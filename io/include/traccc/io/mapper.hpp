/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "traccc/clusterization/component_connection.hpp"
#include "traccc/clusterization/measurement_creation.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/particle.hpp"
#include "traccc/io/csv.hpp"
#include "traccc/io/reader.hpp"
#include "traccc/io/utils.hpp"

namespace traccc {

using hit_id = uint64_t;
using hit_particle_map = std::map<spacepoint, particle>;
using hit_map = std::map<hit_id, spacepoint>;
using hit_cell_map = std::map<spacepoint, std::vector<cell>>;
using cell_particle_map = std::map<cell, particle>;
using measurement_cell_map = std::map<measurement, std::vector<cell>>;
using measurement_particle_map = std::map<measurement, std::vector<particle>>;

hit_particle_map generate_hit_particle_map(size_t event,
                                           const std::string& hits_dir) {
    hit_particle_map result;

    // Read the hits from the relevant event file
    std::string io_hits_file =
        data_directory() + hits_dir + get_event_filename(event, "-hits.csv");

    fatras_hit_reader hreader(
        io_hits_file,
        {"particle_id", "geometry_id", "tx", "ty", "tz", "tt", "tpx", "tpy",
         "tpz", "te", "deltapx", "deltapy", "deltapz", "deltae", "index"});

    csv_fatras_hit iohit;

    while (hreader.read(iohit)) {

        spacepoint sp;
        sp.global = {iohit.tx, iohit.ty, iohit.tz};

        particle ptc;
        ptc.pid = iohit.particle_id;

        result[sp] = ptc;
    }

    return result;
}

hit_map generate_hit_map(size_t event, const std::string& hits_dir) {
    hit_map result;

    // Read the hits from the relevant event file
    std::string io_hits_file =
        data_directory() + hits_dir + get_event_filename(event, "-hits.csv");

    fatras_hit_reader hreader(
        io_hits_file,
        {"particle_id", "geometry_id", "tx", "ty", "tz", "tt", "tpx", "tpy",
         "tpz", "te", "deltapx", "deltapy", "deltapz", "deltae", "index"});

    csv_fatras_hit iohit;

    hit_id hid = 0;
    while (hreader.read(iohit)) {

        spacepoint sp;
        sp.global = {iohit.tx, iohit.ty, iohit.tz};

        result[hid] = sp;

        hid++;
    }

    return result;
}

hit_cell_map generate_hit_cell_map(size_t event, const std::string& cells_dir,
                                   const std::string& hits_dir) {

    hit_cell_map result;

    auto hmap = generate_hit_map(event, hits_dir);

    // Read the cells from the relevant event file
    std::string io_cells_file =
        data_directory() + cells_dir + get_event_filename(event, "-cells.csv");

    cell_reader creader(io_cells_file, {"geometry_id", "hit_id", "cannel0",
                                        "channel1", "activation", "time"});

    csv_cell iocell;

    while (creader.read(iocell)) {
        result[hmap[iocell.hit_id]].push_back(cell{
            iocell.channel0, iocell.channel1, iocell.value, iocell.timestamp});
    }

    return result;
}

cell_particle_map generate_cell_particle_map(size_t event,
                                             const std::string& cells_dir,
                                             const std::string& hits_dir) {

    cell_particle_map result;

    auto h_p_map = generate_hit_particle_map(event, hits_dir);

    auto h_c_map = generate_hit_cell_map(event, cells_dir, hits_dir);

    for (auto const& [hit, ptc] : h_p_map) {
        auto& cells = h_c_map[hit];

        for (auto& c : cells) {
            result[c] = ptc;
        }
    }

    return result;
}

measurement_cell_map generate_measurement_cell_map(
    size_t event, const std::string& detector_file,
    const std::string& cells_dir, vecmem::host_memory_resource& resource) {

    measurement_cell_map result;

    // CCA algorithms
    component_connection cc(resource);
    measurement_creation mt(resource);

    // Read the surface transforms
    auto surface_transforms = read_geometry(detector_file);

    // Read the cells from the relevant event file
    host_cell_container cells_per_event =
        read_cells_from_event(event, cells_dir, surface_transforms, resource);

    for (std::size_t i = 0; i < cells_per_event.size(); ++i) {
        auto module = cells_per_event.at(i).header;

        // The algorithmic code part: start
        cluster_collection clusters_per_module =
            cc(cells_per_event.at(i).items, cells_per_event.at(i).header);
        clusters_per_module.position_from_cell = module.pixel;

        host_measurement_collection measurements_per_module =
            mt(clusters_per_module, module);

        for (std::size_t j = 0; j < clusters_per_module.items.size(); j++) {
            const auto& clus = clusters_per_module.items[j];
            const auto& meas = measurements_per_module[j];

            result[meas] = clus.cells;
        }
    }

    return result;
}

measurement_particle_map generate_measurement_particle_map(
    size_t event, const std::string& detector_file,
    const std::string& cells_dir, const std::string& hits_dir,
    vecmem::host_memory_resource& resource) {

    measurement_particle_map result;

    // generate cell particle map
    auto c_p_map = generate_cell_particle_map(event, cells_dir, hits_dir);

    // generate measurement cell map
    auto m_c_map = generate_measurement_cell_map(event, detector_file,
                                                 cells_dir, resource);

    for (auto const& [meas, cells] : m_c_map) {
        for (const auto& c : cells) {
            result[meas].push_back(c_p_map[c]);
        }
    }

    return result;
}

}  // namespace traccc