/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/io/event_data.hpp"

#include "traccc/io/csv/make_cell_reader.hpp"
#include "traccc/io/csv/make_hit_reader.hpp"
#include "traccc/io/csv/make_measurement_hit_id_reader.hpp"
#include "traccc/io/csv/make_measurement_reader.hpp"
#include "traccc/io/csv/make_particle_reader.hpp"
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_digitization_config.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/utils/particle.hpp"

// Detray include(s).
#include "detray/io/frontend/detector_reader.hpp"

// System include(s).
#include <filesystem>

namespace traccc {

event_data::event_data(const std::string& event_dir, const std::size_t event_id,
                       vecmem::memory_resource& resource,
                       bool use_acts_geom_source, const detector_type* det,
                       data_format format, bool include_silicon_cells)
    : m_event_dir(event_dir), m_event_id(event_id), m_mr(resource) {

    // Currently, we only support csv type for event data
    assert(format == data_format::csv);
    if (format == data_format::csv) {
        setup_csv(use_acts_geom_source, det, include_silicon_cells);
    }
}

void event_data::setup_csv(bool use_acts_geom_source, const detector_type* det,
                           bool include_silicon_cells) {

    /********************
     *  Read Csv files  *
     ********************/

    // CSV IO EDM containers
    std::vector<io::csv::cell> m_cells;
    std::vector<io::csv::hit> m_hits;
    std::vector<io::csv::measurement> m_measurements;
    std::vector<io::csv::measurement_hit_id> m_meas_hit_ids;
    std::vector<io::csv::particle> m_particles;

    if (include_silicon_cells) {
        // Read the cells from the relevant event file
        std::string io_cells_file =
            io::get_absolute_path((std::filesystem::path(m_event_dir) /
                                   std::filesystem::path(io::get_event_filename(
                                       m_event_id, "-cells.csv")))
                                      .native());

        auto creader = io::csv::make_cell_reader(io_cells_file);
        io::csv::cell iocell;

        while (creader.read(iocell)) {
            m_cells.push_back(iocell);
        }
    }

    // Read the hits from the relevant event file
    std::string io_hits_file = io::get_absolute_path(
        (std::filesystem::path(m_event_dir) /
         std::filesystem::path(io::get_event_filename(m_event_id, "-hits.csv")))
            .native());

    auto hreader = io::csv::make_hit_reader(io_hits_file);
    io::csv::hit iohit;

    while (hreader.read(iohit)) {
        m_hits.push_back(iohit);
    }

    // Read the measurements from the relevant event file
    std::string io_measurements_file =
        io::get_absolute_path((std::filesystem::path(m_event_dir) /
                               std::filesystem::path(io::get_event_filename(
                                   m_event_id, "-measurements.csv")))
                                  .native());

    auto mreader = io::csv::make_measurement_reader(io_measurements_file);
    io::csv::measurement iomeas;

    while (mreader.read(iomeas)) {
        m_measurements.push_back(iomeas);
    }

    // Read the measurement hit id from the relevant event file
    std::string io_measurement_hit_id_file =
        io::get_absolute_path((std::filesystem::path(m_event_dir) /
                               std::filesystem::path(io::get_event_filename(
                                   m_event_id, "-measurement-simhit-map.csv")))
                                  .native());

    auto mhid_reader =
        io::csv::make_measurement_hit_id_reader(io_measurement_hit_id_file);
    io::csv::measurement_hit_id mh_id;

    while (mhid_reader.read(mh_id)) {
        m_meas_hit_ids.push_back(mh_id);
    }

    // Read the particles from the relevant event file
    std::string io_particles_file =
        io::get_absolute_path((std::filesystem::path(m_event_dir) /
                               std::filesystem::path(io::get_event_filename(
                                   m_event_id, "-particles_initial.csv")))
                                  .native());

    auto preader = io::csv::make_particle_reader(io_particles_file);
    io::csv::particle ioptc;

    while (preader.read(ioptc)) {
        m_particles.push_back(ioptc);
    }

    // Particle map
    for (const auto& csv_ptc : m_particles) {

        point3 pos{csv_ptc.vx, csv_ptc.vy, csv_ptc.vz};
        vector3 mom{csv_ptc.px, csv_ptc.py, csv_ptc.pz};

        m_particle_map[csv_ptc.particle_id] =
            traccc::particle{csv_ptc.particle_id, csv_ptc.particle_type,
                             csv_ptc.process,     pos,
                             csv_ptc.vt,          mom,
                             csv_ptc.m,           csv_ptc.q};
    }

    /********************
     * Make geom_id map *
     ********************/

    // For Acts data, build a map of acts->detray geometry IDs
    std::map<geometry_id, geometry_id> acts_to_detray_id;
    if (use_acts_geom_source) {
        for (const auto& surface_desc : det->surfaces()) {
            acts_to_detray_id[surface_desc.source] =
                surface_desc.barcode().value();
        }
    }

    /************************************
     * Make measurement to particle map *
     ************************************/

    // When including silicon cells
    if (include_silicon_cells) {

        std::map<io::csv::cell, particle> m_cell_to_particle_map;
        std::map<measurement, std::vector<io::csv::cell>> meas_to_cluster_map;

        for (const auto& csv_cell : m_cells) {

            // Fill the measurement_to_cluster_map
            auto meas_id = csv_cell.measurement_id;
            auto hid = m_meas_hit_ids[meas_id].hit_id;
            iohit = m_hits[hid];

            iomeas = m_measurements[meas_id];
            const auto meas = traccc::io::csv::make_measurement_edm(
                iomeas, &acts_to_detray_id);
            meas_to_cluster_map[meas].push_back(csv_cell);
            m_cell_to_particle_map[csv_cell] =
                m_particle_map[iohit.particle_id];
        }

        // Fill the meas_to_particle_map
        for (auto const& [ms, cluster] : meas_to_cluster_map) {
            for (const auto& cl : cluster) {
                m_meas_to_ptc_map[ms][m_cell_to_particle_map[cl]]++;
            }
        }
    }

    for (const auto& csv_meas : m_measurements) {

        // Hit index
        const auto hid = m_meas_hit_ids[csv_meas.measurement_id].hit_id;

        // Make spacepoint
        iohit = m_hits[hid];
        point3 global_pos{iohit.tx, iohit.ty, iohit.tz};
        point3 global_mom{iohit.tpx, iohit.tpy, iohit.tpz};

        // Make particle
        const auto ptc = m_particle_map[iohit.particle_id];

        // Construct the measurement object.
        traccc::measurement meas;
        if (use_acts_geom_source) {
            meas = traccc::io::csv::make_measurement_edm(csv_meas,
                                                         &acts_to_detray_id);
        } else {
            meas = traccc::io::csv::make_measurement_edm(csv_meas, nullptr);
        }

        // Fill measurement to truth global position and momentum map
        m_meas_to_param_map[meas] = std::make_pair(global_pos, global_mom);

        // Fill particle to measurement map
        m_ptc_to_meas_map[ptc].push_back(meas);

        if (!include_silicon_cells) {
            // Fill measurement to particle map
            m_meas_to_ptc_map[meas][ptc]++;
        }
    }
}

/*
void event_data::fill_cca_result(
    const cluster_container_types::host& cca_clusters,
    const measurement_collection_types::host& cca_measurements) {

    const std::size_t n_cca_clusters = cca_measurements.size();

    std::map<measurement, vecmem::vector<cell>> found_meas_to_cluster_map;

    for (std::size_t i = 0; i < n_cca_clusters; i++) {
        const auto meas = cca_measurements.at(i);
        auto clus = cca_clusters.at(i).items;
        std::sort(clus.begin(), clus.end());
        m_cluster_to_found_meas_map[clus] = meas;
        found_meas_to_cluster_map[meas] = clus;
    }

    for (auto const& [ms, cluster] : found_meas_to_cluster_map) {
        for (const auto& cl : cluster) {
            m_found_meas_to_ptc_map[ms][m_cell_to_particle_map[cl]]++;
        }
    }
}
*/

track_candidate_container_types::host event_data::generate_truth_candidates(
    seed_generator<detector_type>& sg, vecmem::memory_resource& resource) {

    traccc::track_candidate_container_types::host track_candidates(&resource);

    for (auto const& [ptc, measurements] : m_ptc_to_meas_map) {

        const auto& param = m_meas_to_param_map[measurements[0]];
        const free_track_parameters free_param(param.first, 0.f, param.second,
                                               ptc.charge);

        auto seed_params =
            sg(measurements[0].surface_link, free_param,
               detail::particle_from_pdg_number<scalar>(ptc.particle_type));

        // Candidate objects
        vecmem::vector<track_candidate> candidates;
        candidates.reserve(measurements.size());

        for (const auto& meas : measurements) {
            candidates.push_back(meas);
        }

        track_candidates.push_back(std::move(seed_params),
                                   std::move(candidates));
    }

    return track_candidates;
}

}  // namespace traccc