/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

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
using particle_id = uint64_t;
using particle_map = std::map<particle_id, particle>;
using hit_particle_map = std::map<spacepoint, particle>;
using hit_map = std::map<hit_id, spacepoint>;
using hit_cell_map = std::map<spacepoint, std::vector<cell>>;
using cell_particle_map = std::map<cell, particle>;
using measurement_cell_map = std::map<measurement, vecmem::vector<cell>>;
using measurement_particle_map =
    std::map<measurement, std::map<particle, uint64_t>>;

class event_map {

    public:
    event_map() = delete;

    event_map(std::size_t event, const std::string& detector_file,
              const std::string& cell_dir, const std::string& hit_dir,
              const std::string particle_dir, vecmem::memory_resource& resource)
        : m_event(event),
          m_detector_file(detector_file),
          m_cell_dir(cell_dir),
          m_hit_dir(hit_dir),
          m_particle_dir(particle_dir),
          m_mr(resource) {

        // truth map
        m_hit_map = generate_hit_map();
        m_particle_map = generate_particle_map();
        m_hit_particle_map = generate_hit_particle_map(m_particle_map);
        m_hit_cell_map = generate_hit_cell_map(m_hit_map);
        m_cell_particle_map =
            generate_cell_particle_map(m_hit_particle_map, m_hit_cell_map);
        // reconstructed map
        m_measurement_cell_map = generate_measurement_cell_map();
        m_measurement_particle_map = generate_measurement_particle_map(
            m_cell_particle_map, m_measurement_cell_map);
    }

    event_map(std::size_t event, const std::string& detector_file,
              const std::string& hit_dir, const std::string particle_dir,
              vecmem::memory_resource& resource)
        : m_event(event),
          m_detector_file(detector_file),
          m_cell_dir(""),
          m_hit_dir(hit_dir),
          m_particle_dir(particle_dir),
          m_mr(resource) {

        m_hit_map = generate_hit_map();
        m_particle_map = generate_particle_map();
        m_hit_particle_map = generate_hit_particle_map(m_particle_map);
        m_measurement_particle_map =
            generate_measurement_particle_map(m_hit_particle_map);
    }

    hit_map& get_hit_map() { return m_hit_map; }

    particle_map& get_particle_map() { return m_particle_map; }

    hit_particle_map& get_hit_particle_map() { return m_hit_particle_map; }

    hit_cell_map& get_hit_cell_map() { return m_hit_cell_map; }

    cell_particle_map& get_cell_particle_map() { return m_cell_particle_map; }

    measurement_cell_map& get_measurement_cell_map() {
        return m_measurement_cell_map;
    }

    measurement_particle_map& get_measurement_particle_map() {
        return m_measurement_particle_map;
    }

    particle_map generate_particle_map() {

        particle_map result;

        // Read the particles from the relevant event file
        std::string io_particles_file =
            data_directory() + m_particle_dir +
            get_event_filename(m_event, "-particles_initial.csv");

        fatras_particle_reader preader(
            io_particles_file, {"particle_id", "particle_type", "process", "vx",
                                "vy", "vz", "vt", "px", "py", "pz", "m", "q"});

        csv_particle ioptc;

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

    hit_particle_map generate_hit_particle_map(particle_map& p_map) {
        hit_particle_map result;

        // Read the hits from the relevant event file
        std::string io_hits_file = data_directory() + m_hit_dir +
                                   get_event_filename(m_event, "-hits.csv");

        fatras_hit_reader hreader(
            io_hits_file,
            {"particle_id", "geometry_id", "tx", "ty", "tz", "tt", "tpx", "tpy",
             "tpz", "te", "deltapx", "deltapy", "deltapz", "deltae", "index"});

        csv_fatras_hit iohit;

        while (hreader.read(iohit)) {

            spacepoint sp;
            sp.global = {iohit.tx, iohit.ty, iohit.tz};

            particle ptc = p_map[iohit.particle_id];

            result[sp] = ptc;
        }

        return result;
    }

    hit_map generate_hit_map() {
        hit_map result;

        // Read the hits from the relevant event file
        std::string io_hits_file = data_directory() + m_hit_dir +
                                   get_event_filename(m_event, "-hits.csv");

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

    hit_cell_map generate_hit_cell_map(hit_map& h_map) {

        hit_cell_map result;

        // Read the cells from the relevant event file
        std::string io_cells_file = data_directory() + m_cell_dir +
                                    get_event_filename(m_event, "-cells.csv");

        cell_reader creader(io_cells_file, {"geometry_id", "hit_id", "cannel0",
                                            "channel1", "activation", "time"});

        csv_cell iocell;

        while (creader.read(iocell)) {
            result[h_map[iocell.hit_id]].push_back(
                cell{iocell.channel0, iocell.channel1, iocell.value,
                     iocell.timestamp});
        }

        return result;
    }

    cell_particle_map generate_cell_particle_map(hit_particle_map& h_p_map,
                                                 hit_cell_map& h_c_map) {

        cell_particle_map result;

        for (auto const& [hit, ptc] : h_p_map) {
            auto& cells = h_c_map[hit];

            for (auto& c : cells) {
                result[c] = ptc;
            }
        }

        return result;
    }

    measurement_cell_map generate_measurement_cell_map() {

        measurement_cell_map result;

        // CCA algorithms
        component_connection cc(m_mr.get());
        measurement_creation mt(m_mr.get());

        // Read the surface transforms
        auto surface_transforms = read_geometry(m_detector_file);

        // Read the cells from the relevant event file
        host_cell_container cells_per_event = read_cells_from_event(
            m_event, m_cell_dir, surface_transforms, m_mr.get());

        for (std::size_t i = 0; i < cells_per_event.size(); ++i) {
            auto module = cells_per_event.at(i).header;

            // The algorithmic code part: start
            host_cluster_container clusters =
                cc(cells_per_event.at(i).items, cells_per_event.at(i).header);

            for (auto& cl_id : clusters.get_headers()) {
                cl_id.pixel = module.pixel;
            }
            
            host_measurement_collection measurements_per_module =
                mt(clusters, module);

            for (std::size_t j = 0; j < clusters.size(); j++) {
                const auto& clus = clusters.at(j).items;
                const auto& meas = measurements_per_module[j];

                result[meas] = clus;
            }
        }

        return result;
    }

    measurement_particle_map generate_measurement_particle_map(
        cell_particle_map& c_p_map, measurement_cell_map& m_c_map) {

        measurement_particle_map result;

        for (auto const& [meas, cells] : m_c_map) {
            for (const auto& c : cells) {
                result[meas][c_p_map[c]]++;
            }
        }

        return result;
    }

    measurement_particle_map generate_measurement_particle_map(
        hit_particle_map& h_p_map) {

        measurement_particle_map result;

        // Read the surface transforms
        auto surface_transforms = read_geometry(m_detector_file);

        // Read the spacepoints from the relevant event file
        host_spacepoint_container spacepoints_per_event =
            read_spacepoints_from_event(m_event, m_hit_dir, surface_transforms,
                                        m_mr.get());

        for (std::size_t i = 0; i < spacepoints_per_event.size(); ++i) {
            const auto& spacepoints_per_module =
                spacepoints_per_event.at(i).items;

            for (const auto& hit : spacepoints_per_module) {
                const auto& meas = hit.meas;

                spacepoint new_hit;
                new_hit.global = hit.global;

                const auto& ptc = h_p_map[new_hit];
                result[meas][ptc]++;
            }
        }

        return result;
    }

    private:
    const std::size_t m_event;
    const std::string m_detector_file;
    const std::string m_cell_dir;
    const std::string m_hit_dir;
    const std::string m_particle_dir;
    std::reference_wrapper<vecmem::memory_resource> m_mr;

    particle_map m_particle_map;
    hit_particle_map m_hit_particle_map;
    hit_map m_hit_map;
    hit_cell_map m_hit_cell_map;
    cell_particle_map m_cell_particle_map;
    measurement_cell_map m_measurement_cell_map;
    measurement_particle_map m_measurement_particle_map;
};

}  // namespace traccc