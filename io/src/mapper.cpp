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
#include <cassert>
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
                                   vecmem::memory_resource& resource,
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

    // Helper lambda for setting up SoA cells.
    auto assign = [](auto dest, const io::csv::cell& src,
                     unsigned int module_idx) {
        dest.channel0() = src.channel0;
        dest.channel1() = src.channel1;
        dest.activation() = src.value;
        dest.time() = src.timestamp;
        dest.module_index() = module_idx;
    };

    while (creader.read(iocell)) {
        unsigned int link = 0;
        auto it = link_map.find(iocell.geometry_id);
        if (it != link_map.end()) {
            link = (*it).second;
        }
        auto pair = result.insert({hmap[iocell.measurement_id], {resource}});
        edm::silicon_cell_collection::host& cells = pair.first->second;
        const std::size_t cellIndex = cells.size();
        cells.resize(cellIndex + 1);
        assign(cells.at(cellIndex), iocell, link);
    }
    return result;
}

particle_cell_map generate_particle_cell_map(std::size_t event,
                                             const std::string& cells_dir,
                                             const std::string& hits_dir,
                                             const std::string& particle_dir,
                                             vecmem::memory_resource& resource,
                                             const geoId_link_map& link_map) {

    particle_cell_map result;

    auto h_p_map =
        generate_hit_particle_map(event, hits_dir, particle_dir, link_map);

    auto h_c_map =
        generate_hit_cell_map(event, cells_dir, hits_dir, resource, link_map);

    for (auto const& [hit, ptc] : h_p_map) {
        auto pair1 = h_c_map.insert({hit, {resource}});
        edm::silicon_cell_collection::host& cells1 = pair1.first->second;
        auto pair2 = result.insert({ptc, {resource}});
        edm::silicon_cell_collection::host& cells2 = pair2.first->second;

        for (std::size_t i = 0; i < cells1.size(); ++i) {
            const std::size_t cellIndex = cells2.size();
            cells2.resize(cellIndex + 1);
            cells2.at(cellIndex) = cells1.at(i);
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
    edm::silicon_cell_collection::host cells(resource);
    io::read_cells(cells, event, cells_dir, &dd, traccc::data_format::csv,
                   false);

    auto cells_data = vecmem::get_data(cells);
    auto clusters = cc(cells_data);
    auto clusters_data = vecmem::get_data(clusters);
    auto measurements = mc(cells_data, clusters_data, dd_data);

    assert(measurements.size() == clusters.size());
    for (std::size_t i = 0; i < measurements.size(); ++i) {

        edm::silicon_cell_collection::host cells_for_measurement{resource};
        cells_for_measurement.reserve(clusters.cell_indices().at(i).size());
        for (unsigned int cell_idx : clusters.cell_indices().at(i)) {
            const std::size_t measCellIndex = cells_for_measurement.size();
            cells_for_measurement.resize(measCellIndex + 1);
            cells_for_measurement.at(measCellIndex) = cells.at(cell_idx);
        }

        auto pair = result.insert({measurements.at(i), cells_for_measurement});
        (void)pair;
        assert(pair.second);
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
    for (unsigned int i = 0; i < dd.acts_geometry_id().size(); ++i) {
        link_map[dd.geometry_id()[i].value()] = i;
    }

    // generate cell particle map
    auto p_c_map = generate_particle_cell_map(event, cells_dir, hits_dir,
                                              particle_dir, resource, link_map);

    // Loop over the map associating cells with measurements.
    for (auto const& [meas, mCells] : m_c_map) {
        // Loop over the map associating particles with cells.
        for (auto const& [ptc, pCells] : p_c_map) {
            // Check how many cells are shared between the two.
            for (std::size_t mCellIndex = 0; mCellIndex < mCells.size();
                 ++mCellIndex) {
                for (std::size_t pCellIndex = 0; pCellIndex < pCells.size();
                     ++pCellIndex) {
                    if (mCells.at(mCellIndex) == pCells.at(pCellIndex)) {

                        result[meas][ptc]++;
                    }
                }
            }
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
        link_map[dd.geometry_id()[i].value()] = i;
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
