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
#include "csv/make_measurement_hit_id_reader.hpp"
#include "csv/make_particle_reader.hpp"
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_digitization_config.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/read_spacepoints_alt.hpp"
#include "traccc/io/utils.hpp"

// Project include(s).
#include "traccc/clusterization/component_connection.hpp"
#include "traccc/clusterization/measurement_creation.hpp"

namespace traccc {

particle_map generate_particle_map(std::size_t event,
                                   const std::string& particle_dir) {

    particle_map result;

    // Read the particles from the relevant event file
    std::string io_particles_file =
        io::data_directory() + particle_dir +
        io::get_event_filename(event, "-particles_initial.csv");

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
    std::string io_hits_file = io::data_directory() + hits_dir +
                               io::get_event_filename(event, "-hits.csv");

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
    std::string io_hits_file = io::data_directory() + hits_dir +
                               io::get_event_filename(event, "-hits.csv");

    auto hreader = io::csv::make_hit_reader(io_hits_file);

    io::csv::hit iohit;

    // Read the hits from the relevant event file
    std::string io_meas_hit_id_file =
        io::data_directory() + hits_dir +
        io::get_event_filename(event, "-measurement-simhit-map.csv");

    auto mhid_reader =
        io::csv::make_measurement_hit_id_reader(io_meas_hit_id_file);

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
    std::string io_cells_file = io::data_directory() + cells_dir +
                                io::get_event_filename(event, "-cells.csv");

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

std::tuple<measurement_cell_map, cell_module_collection_types::host>
generate_measurement_cell_map(std::size_t event,
                              const std::string& detector_file,
                              const std::string& digi_config_file,
                              const std::string& cells_dir,
                              vecmem::memory_resource& resource) {

    measurement_cell_map result;

    // CCA algorithms
    component_connection cc(resource);
    measurement_creation mc(resource);

    // Read the surface transforms
    auto surface_transforms = io::read_geometry(detector_file);

    // Read the digitization configuration file
    auto digi_cfg = io::read_digitization_config(digi_config_file);

    // Read the cells from the relevant event file
    auto readOut = io::read_cells(event, cells_dir, traccc::data_format::csv,
                                  &surface_transforms, &digi_cfg, &resource);
    cell_collection_types::host& cells_per_event = readOut.cells;
    cell_module_collection_types::host& modules_per_event = readOut.modules;

    auto clusters_per_event = cc(cells_per_event);
    auto measurements_per_event = mc(clusters_per_event, modules_per_event);

    assert(measurements_per_event.size() == clusters_per_event.size());
    for (unsigned int i = 0; i < measurements_per_event.size(); ++i) {
        const auto& clus = clusters_per_event.get_items()[i];

        result[measurements_per_event[i]] = clus;
    }

    return {result, modules_per_event};
}

measurement_particle_map generate_measurement_particle_map(
    std::size_t event, const std::string& detector_file,
    const std::string& digi_config_file, const std::string& cells_dir,
    const std::string& hits_dir, const std::string& particle_dir,
    vecmem::memory_resource& resource) {

    measurement_particle_map result;

    // generate measurement cell map
    auto gen_m_c_map = generate_measurement_cell_map(
        event, detector_file, digi_config_file, cells_dir, resource);
    auto& m_c_map = std::get<0>(gen_m_c_map);
    auto& modules = std::get<1>(gen_m_c_map);

    // generate geometry_id link map
    geoId_link_map link_map;

    for (unsigned int i = 0; i < modules.size(); ++i) {
        link_map[modules[i].module] = i;
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
    std::size_t event, const std::string& detector_file,
    const std::string& hits_dir, const std::string& particle_dir,
    vecmem::memory_resource& resource) {

    measurement_particle_map result;

    // Read the surface transforms
    auto surface_transforms = io::read_geometry(detector_file);

    // Read the spacepoints from the relevant event file
    auto readOut =
        io::read_spacepoints_alt(event, hits_dir, surface_transforms,
                                 traccc::data_format::csv, &resource);
    spacepoint_collection_types::host& spacepoints_per_event =
        readOut.spacepoints;
    cell_module_collection_types::host& modules = readOut.modules;

    geoId_link_map link_map;

    for (unsigned int i = 0; i < modules.size(); ++i) {
        link_map[modules[i].module] = i;
    }

    auto h_p_map =
        generate_hit_particle_map(event, hits_dir, particle_dir, link_map);

    for (const auto& hit : spacepoints_per_event) {
        const auto& meas = hit.meas;

        spacepoint new_hit;
        new_hit.global = hit.global;

        const auto& ptc = h_p_map[new_hit];
        result[meas][ptc]++;
    }

    return result;
}

}  // namespace traccc
