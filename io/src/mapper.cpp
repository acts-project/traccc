/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/io/mapper.hpp"

#include "traccc/io/csv/make_cell_reader.hpp"
#include "traccc/io/csv/make_hit_reader.hpp"
#include "traccc/io/csv/make_measurement_hit_id_reader.hpp"
#include "traccc/io/csv/make_particle_reader.hpp"
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_detector_description.hpp"
#include "traccc/io/read_digitization_config.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/read_spacepoints.hpp"
#include "traccc/io/utils.hpp"

// Project include(s).
#include "traccc/clusterization/measurement_creation_algorithm.hpp"
#include "traccc/clusterization/sparse_ccl_algorithm.hpp"

// System include(s).
#include <filesystem>

namespace traccc {

particle_map generate_particle_map(std::size_t event,
                                   const std::string& particle_dir) {

    particle_map result;

    // Read the particles from the relevant event file
    std::string io_particles_file =
        io::get_absolute_path((std::filesystem::path(particle_dir) /
                               std::filesystem::path(io::get_event_filename(
                                   event, "-particles_initial.csv")))
                                  .native());

    auto preader = io::csv::make_particle_reader(io_particles_file);

    io::csv::particle ioptc;

    while (preader.read(ioptc)) {
        point3 pos{ioptc.vx, ioptc.vy, ioptc.vz};
        vector3 mom{ioptc.px, ioptc.py, ioptc.pz};

        result[ioptc.particle_id] =
            particle{ioptc.particle_id, ioptc.particle_type,
                     ioptc.process,     pos,
                     ioptc.vt,          mom,
                     ioptc.m,           ioptc.q};
    }

    return result;
}

hit_particle_map generate_hit_particle_map(std::size_t event,
                                           const std::string& hits_dir,
                                           const std::string& particle_dir,
                                           const geoId_link_map& link_map) {
    hit_particle_map result;

    auto pmap = generate_particle_map(event, particle_dir);

    // Read the hits from the relevant event file
    std::string io_hits_file = io::get_absolute_path(
        (std::filesystem::path(hits_dir) /
         std::filesystem::path(io::get_event_filename(event, "-hits.csv")))
            .native());

    auto hreader = io::csv::make_hit_reader(io_hits_file);

    io::csv::hit iohit;

    while (hreader.read(iohit)) {

        spacepoint sp;
        sp.global = {iohit.tx, iohit.ty, iohit.tz};

        unsigned int link = 0;
        auto it = link_map.find(iohit.geometry_id);
        if (it != link_map.end()) {
            link = (*it).second;
        }

        sp.meas.module_link = link;

        particle ptc = pmap[iohit.particle_id];

        result[sp] = ptc;
    }

    return result;
}

hit_map generate_hit_map(std::size_t event, const std::string& hits_dir) {
    hit_map result;

    // Read the hits from the relevant event file
    std::string io_hits_file = io::get_absolute_path(
        (std::filesystem::path(hits_dir) /
         std::filesystem::path(io::get_event_filename(event, "-hits.csv")))
            .native());

    auto hreader = io::csv::make_hit_reader(io_hits_file);

    io::csv::hit iohit;

    // Read the hits from the relevant event file
    std::string io_measurement_hit_id_file =
        io::get_absolute_path((std::filesystem::path(hits_dir) /
                               std::filesystem::path(io::get_event_filename(
                                   event, "-measurement-simhit-map.csv")))
                                  .native());

    auto mhid_reader =
        io::csv::make_measurement_hit_id_reader(io_measurement_hit_id_file);

    io::csv::measurement_hit_id mh_id;

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

hit_cell_map generate_hit_cell_map(std::size_t event,
                                   const std::string& cells_dir,
                                   const std::string& hits_dir,
                                   const geoId_link_map& link_map) {

    hit_cell_map result;

    auto hmap = generate_hit_map(event, hits_dir);

    // Read the cells from the relevant event file
    std::string io_cells_file = io::get_absolute_path(
        (std::filesystem::path(cells_dir) /
         std::filesystem::path(io::get_event_filename(event, "-cells.csv")))
            .native());

    auto creader = io::csv::make_cell_reader(io_cells_file);

    io::csv::cell iocell;

    while (creader.read(iocell)) {
        unsigned int link = 0;
        auto it = link_map.find(iocell.geometry_id);
        if (it != link_map.end()) {
            link = (*it).second;
        }
        result[hmap[iocell.hit_id]].push_back(
            cell{iocell.channel0, iocell.channel1, iocell.value,
                 iocell.timestamp, link});
    }
    return result;
}

cell_particle_map generate_cell_particle_map(std::size_t event,
                                             const std::string& cells_dir,
                                             const std::string& hits_dir,
                                             const std::string& particle_dir,
                                             const geoId_link_map& link_map) {

    cell_particle_map result;

    auto h_p_map =
        generate_hit_particle_map(event, hits_dir, particle_dir, link_map);

    auto h_c_map = generate_hit_cell_map(event, cells_dir, hits_dir, link_map);

    for (auto const& [hit, ptc] : h_p_map) {
        auto& cells = h_c_map[hit];

        for (auto& c : cells) {
            result[c] = ptc;
        }
    }

    return result;
}

measurement_cell_map generate_measurement_cell_map(
    std::size_t event, const std::string& cells_dir,
    const silicon_detector_description::host& dd,
    vecmem::memory_resource& resource) {

    measurement_cell_map result;

    // CCA algorithms
    host::sparse_ccl_algorithm cc(resource);
    host::measurement_creation_algorithm mc(resource);

    // Construct a detector description data object.
    silicon_detector_description::const_data dd_data{vecmem::get_data(dd)};

    // Read the cells from the relevant event file
    cell_collection_types::host cells(&resource);
    io::read_cells(cells, event, cells_dir, &dd, traccc::data_format::csv,
                   false);

    auto clusters_per_event = cc(vecmem::get_data(cells));
    auto clusters_data = traccc::get_data(clusters_per_event);
    auto measurements_per_event = mc(clusters_data, dd_data);

    assert(measurements_per_event.size() == clusters_per_event.size());
    for (unsigned int i = 0; i < measurements_per_event.size(); ++i) {
        const auto& clus = clusters_per_event.get_items()[i];

        result[measurements_per_event[i]] = clus;
    }

    return result;
}

measurement_particle_map generate_measurement_particle_map(
    std::size_t event, const std::string& cells_dir,
    const std::string& hits_dir, const std::string& particle_dir,
    const silicon_detector_description::host& dd,
    vecmem::memory_resource& resource) {

    measurement_particle_map result;

    // generate measurement cell map
    auto m_c_map =
        generate_measurement_cell_map(event, cells_dir, dd, resource);

    // generate geometry_id link map
    geoId_link_map link_map;
    for (unsigned int i = 0; i < dd.geometry_id().size(); ++i) {
        link_map[dd.surface_link()[i].value()] = i;
    }

    // generate cell particle map
    auto c_p_map = generate_cell_particle_map(event, cells_dir, hits_dir,
                                              particle_dir, link_map);

    for (auto const& [meas, cells] : m_c_map) {
        for (const auto& c : cells) {
            result[meas][c_p_map[c]]++;
        }
    }

    return result;
}

measurement_particle_map generate_measurement_particle_map(
    std::size_t event, const std::string& hits_dir,
    const std::string& particle_dir,
    const silicon_detector_description::host& dd,
    vecmem::memory_resource& resource) {

    measurement_particle_map result;

    // Read the spacepoints from the relevant event file
    spacepoint_collection_types::host spacepoints{&resource};
    io::read_spacepoints(spacepoints, event, hits_dir, &dd,
                         traccc::data_format::csv);

    geoId_link_map link_map;
    for (std::size_t i = 0; i < dd.size(); ++i) {
        link_map[dd.surface_link()[i].value()] = i;
    }

    auto h_p_map =
        generate_hit_particle_map(event, hits_dir, particle_dir, link_map);

    for (const auto& hit : spacepoints) {
        const auto& meas = hit.meas;

        spacepoint new_hit;
        new_hit.global = hit.global;

        const auto& ptc = h_p_map[new_hit];
        result[meas][ptc]++;
    }

    return result;
}

}  // namespace traccc
