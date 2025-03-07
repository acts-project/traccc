/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/utils/event_data.hpp"

#include "traccc/io/csv/make_cell_reader.hpp"
#include "traccc/io/csv/make_hit_reader.hpp"
#include "traccc/io/csv/make_measurement_edm.hpp"
#include "traccc/io/csv/make_measurement_hit_id_reader.hpp"
#include "traccc/io/csv/make_measurement_reader.hpp"
#include "traccc/io/csv/make_particle_reader.hpp"
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_digitization_config.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/utils/particle.hpp"

// Detray include(s).
#include <detray/io/frontend/detector_reader.hpp>

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
    std::vector<io::csv::cell> csv_cells;
    std::vector<io::csv::hit> csv_hits;
    std::vector<io::csv::measurement> csv_measurements;
    std::vector<io::csv::measurement_hit_id> csv_meas_hit_ids;
    std::vector<io::csv::particle> csv_particles;

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
            csv_cells.push_back(iocell);
        }
    }

    // Read the hits from the relevant event file
    std::string io_hits_file = io::get_absolute_path(
        (std::filesystem::path(m_event_dir) /
         std::filesystem::path(io::get_event_filename(m_event_id, "-hits.csv")))
            .native());

    auto hreader = io::csv::make_hit_reader(io_hits_file);
    {
        io::csv::hit iohit;
        while (hreader.read(iohit)) {
            csv_hits.push_back(iohit);
        }
    }

    // Read the measurements from the relevant event file
    std::string io_measurements_file =
        io::get_absolute_path((std::filesystem::path(m_event_dir) /
                               std::filesystem::path(io::get_event_filename(
                                   m_event_id, "-measurements.csv")))
                                  .native());

    auto mreader = io::csv::make_measurement_reader(io_measurements_file);
    {
        io::csv::measurement iomeas;
        while (mreader.read(iomeas)) {
            csv_measurements.push_back(iomeas);
        }
    }

    // Read the measurement hit id from the relevant event file
    std::string io_measurement_hit_id_file =
        io::get_absolute_path((std::filesystem::path(m_event_dir) /
                               std::filesystem::path(io::get_event_filename(
                                   m_event_id, "-measurement-simhit-map.csv")))
                                  .native());

    auto mhid_reader =
        io::csv::make_measurement_hit_id_reader(io_measurement_hit_id_file);
    {
        io::csv::measurement_hit_id mh_id;
        while (mhid_reader.read(mh_id)) {
            csv_meas_hit_ids.push_back(mh_id);
        }
    }

    // Read the particles from the relevant event file
    std::string io_particles_file =
        io::get_absolute_path((std::filesystem::path(m_event_dir) /
                               std::filesystem::path(io::get_event_filename(
                                   m_event_id, "-particles_initial.csv")))
                                  .native());

    auto preader = io::csv::make_particle_reader(io_particles_file);
    {
        io::csv::particle ioptc;
        while (preader.read(ioptc)) {
            csv_particles.push_back(ioptc);
        }
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

    /********************
     * Make EDM maps    *
     ********************/

    // Measurement map
    for (const auto& iomeas : csv_measurements) {
        // Construct the measurement object.
        traccc::measurement meas;
        if (use_acts_geom_source) {
            meas = traccc::io::csv::make_measurement_edm(iomeas,
                                                         &acts_to_detray_id);
        } else {
            meas = traccc::io::csv::make_measurement_edm(iomeas, nullptr);
        }
        m_measurement_map[iomeas.measurement_id] = meas;
    }

    // Particle map
    for (const auto& ioptc : csv_particles) {

        point3 pos{ioptc.vx, ioptc.vy, ioptc.vz};
        vector3 mom{ioptc.px, ioptc.py, ioptc.pz};

        m_particle_map[ioptc.particle_id] =
            traccc::particle{ioptc.particle_id, ioptc.particle_type,
                             ioptc.process,     pos,
                             ioptc.vt,          mom,
                             ioptc.m,           ioptc.q};
    }

    /************************************
     * Make measurement to particle map *
     ************************************/

    // When including silicon cells
    if (include_silicon_cells) {

        std::map<measurement, std::vector<io::csv::cell>> meas_to_cluster_map;

        for (const auto& iocell : csv_cells) {

            // Fill the measurement_to_cluster_map
            auto meas_id = iocell.measurement_id;
            auto hid = csv_meas_hit_ids[meas_id].hit_id;
            const auto& iohit = csv_hits[hid];

            const auto meas = m_measurement_map.at(meas_id);
            meas_to_cluster_map[meas].push_back(iocell);

            const auto& ptc = m_particle_map.at(iohit.particle_id);
            m_cell_to_particle_map[iocell] = ptc;
        }

        // Fill the meas_to_particle_map
        for (auto const& [ms, cluster] : meas_to_cluster_map) {
            for (const auto& cell : cluster) {
                const auto& ptc = m_cell_to_particle_map.at(cell);
                m_meas_to_ptc_map[ms][ptc]++;
            }
        }
    }

    for (const auto& iomeas : csv_measurements) {

        // Hit index
        const auto hid = csv_meas_hit_ids.at(iomeas.measurement_id).hit_id;

        // Make spacepoint
        const auto& iohit = csv_hits.at(hid);
        point3 global_pos{iohit.tx, iohit.ty, iohit.tz};
        point3 global_mom{iohit.tpx, iohit.tpy, iohit.tpz};

        // Make particle
        const auto& ptc = m_particle_map.at(iohit.particle_id);

        // Construct the measurement object.
        const traccc::measurement& meas =
            m_measurement_map.at(iomeas.measurement_id);

        // Fill measurement to truth global position and momentum map
        m_meas_to_param_map[meas] = std::make_pair(global_pos, global_mom);

        // Fill particle to measurement map
        m_ptc_to_meas_map[ptc].push_back(meas);

        if (!include_silicon_cells) {
            auto insert_return = m_meas_to_ptc_map.insert({meas, {}});
            if (insert_return.second == false) {
                throw std::runtime_error(
                    "The new measurement should not exist in the "
                    "measurement-to-particle map");
            }
            // Each measurement is created by a single particle unless we use
            // the clusterization results
            (*(insert_return.first)).second[ptc] = 1u;
        }
    }
}

void event_data::fill_cca_result(
    const edm::silicon_cell_collection::host& cells,
    const edm::silicon_cluster_collection::host& cca_clusters,
    const measurement_collection_types::host& cca_measurements,
    const silicon_detector_description::host& dd) {

    const std::size_t n_cca_clusters = cca_measurements.size();

    std::map<measurement, std::vector<io::csv::cell>> found_meas_to_cluster_map;

    for (std::size_t i = 0; i < n_cca_clusters; i++) {
        const auto& meas = cca_measurements.at(i);
        const auto cluster = cca_clusters[i];

        std::vector<io::csv::cell> iocells;
        for (const unsigned int cell_idx : cluster.cell_indices()) {

            const auto cell = cells.at(cell_idx);
            io::csv::cell iocell{dd.acts_geometry_id().at(cell.module_index()),
                                 0u,
                                 cell.channel0(),
                                 cell.channel1(),
                                 static_cast<float>(cell.time()),
                                 static_cast<float>(cell.activation())};

            iocells.push_back(iocell);
        }
        found_meas_to_cluster_map[meas] = iocells;
    }

    for (auto const& [ms, cluster] : found_meas_to_cluster_map) {

        std::map<uint64_t, std::size_t> meas_counts;

        // Cells from CCL
        for (const auto& cell1 : cluster) {

            // Cells from truth [cell, particle] map
            for (auto const& [cell2, ptc] : m_cell_to_particle_map) {
                // Increase the particle number if the cell is the same
                if (cell1.geometry_id == cell2.geometry_id &&
                    cell1.channel0 == cell2.channel0 &&
                    cell1.channel1 == cell2.channel1) {
                    m_found_meas_to_ptc_map[ms][ptc]++;
                    meas_counts[cell2.measurement_id]++;
                }
            }
        }

        // Find most contributing measurement and its corresponding hit
        using pair_type = decltype(meas_counts)::value_type;
        auto pr =
            std::max_element(std::begin(meas_counts), std::end(meas_counts),
                             [](const pair_type& p1, const pair_type& p2) {
                                 return p1.second < p2.second;
                             });

        const auto& meas_id = pr->first;
        m_found_meas_to_param_map[ms] =
            m_meas_to_param_map[m_measurement_map[meas_id]];
    }
}

track_candidate_container_types::host event_data::generate_truth_candidates(
    seed_generator<detector_type>& sg, vecmem::memory_resource& resource) {

    traccc::track_candidate_container_types::host track_candidates(&resource);

    for (auto const& [ptc, measurements] : m_ptc_to_meas_map) {

        const auto& param = m_meas_to_param_map.at(measurements[0]);
        const free_track_parameters<> free_param(param.first, 0.f, param.second,
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

        // Track quality set empty
        track_candidates.push_back(
            finding_result{seed_params, track_quality{0.f, 0.f, 0u}},
            std::move(candidates));
    }

    return track_candidates;
}

}  // namespace traccc
