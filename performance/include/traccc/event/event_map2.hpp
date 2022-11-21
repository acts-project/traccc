/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/io/csv.hpp"
#include "traccc/io/utils.hpp"

namespace traccc {

template <typename detector_type>
struct event_map2 {

    using point3 = typename detector_type::point3;
    using vector3 = typename detector_type::vector3;

    event_map2(const detector_type& det, size_t event,
               const std::string& hit_dir = "",
               const std::string& measurement_dir = "",
               const std::string& particle_dir = "")
        : m_detector(std::make_unique<detector_type>(det)) {

        std::string io_meas_hit_id_file =
            data_directory() + hit_dir +
            get_event_filename(event, "-measurement-simhit-map.csv");

        std::string io_particle_file =
            data_directory() + particle_dir +
            get_event_filename(event, "-particles.csv");

        std::string io_hit_file =
            data_directory() + hit_dir + get_event_filename(event, "-hits.csv");

        std::string io_measurement_file =
            data_directory() + measurement_dir +
            get_event_filename(event, "-measurements.csv");

        meas_hit_id_reader mhid_reader(io_meas_hit_id_file,
                                       {"measurement_id", "hit_id"});

        particle_reader preader(
            io_particle_file, {"particle_id", "particle_type", "process", "vx",
                               "vy", "vz", "vt", "px", "py", "pz", "m", "q"});

        fatras_hit_reader hreader(
            io_hit_file,
            {"particle_id", "geometry_id", "tx", "ty", "tz", "tt", "tpx", "tpy",
             "tpz", "te", "deltapx", "deltapy", "deltapz", "deltae", "index"});

        traccc::measurement_reader mreader(
            io_measurement_file,
            {"measurement_id", "geometry_id", "local_key", "local0", "local1",
             "phi", "theta", "time", "var_local0", "var_local1", "var_phi",
             "var_theta", "var_time"});

        csv_meas_hit_id io_mh_id;
        while (mhid_reader.read(io_mh_id)) {
            m_meas_hit_ids.push_back(io_mh_id);
        }

        csv_particle io_particle;
        while (preader.read(io_particle)) {
            m_particles.push_back(io_particle);
        }

        csv_fatras_hit io_hit;
        while (hreader.read(io_hit)) {
            m_hits.push_back(io_hit);
        }

        csv_measurement io_measurement;
        while (mreader.read(io_measurement)) {
            m_measurements.push_back(io_measurement);
        }

        // Check if the size of measurements is equal to measurement-simhit-map
        assert(m_measurements.size() == m_meas_hit_ids.size());
    }

    std::size_t measurement_id_to_particle_id(std::size_t m_id) {
        // Hit index
        const auto h_id = m_meas_hit_ids[m_id].hit_id;

        // Hit objects
        const auto hit = m_hits[h_id];

        // Return Particle index
        return hit.particle_id;
    }

    template <typename seed_generator_t>
    track_candidates_container_types::host generate_truth_candidates(
        seed_generator_t& sg, vecmem::memory_resource& resource) {

        traccc::track_candidates_container_types::host track_candidates(
            &resource);

        std::map<std::size_t, std::vector<std::size_t>> ptc_id_to_meas_ids;

        for (std::size_t m_id = 0; m_id < m_measurements.size(); m_id++) {

            // Hit index
            const auto h_id = m_meas_hit_ids[m_id].hit_id;

            // Hit objects
            const auto hit = m_hits[h_id];

            // Particle index
            const auto p_id = hit.particle_id;

            ptc_id_to_meas_ids[p_id].push_back(m_id);
        }

        for (auto const& [ptc_id, meas_ids] : ptc_id_to_meas_ids) {

            // Particle obejcts
            const auto csv_ptc = m_particles[ptc_id];

            point3 pos{csv_ptc.vx, csv_ptc.vy, csv_ptc.vz};
            vector3 mom{csv_ptc.px, csv_ptc.py, csv_ptc.pz};

            // Make a seed parameter
            free_track_parameters vertex(pos, csv_ptc.vt, mom, csv_ptc.q);

            auto seed_params = sg(vertex);

            // Candidate objects
            vecmem::vector<track_candidate> candidates;

            for (const auto& meas_id : meas_ids) {
                const auto& csv_meas = m_measurements[meas_id];

                point2 local{csv_meas.local0, csv_meas.local1};
                variance2 var{csv_meas.var_local0, csv_meas.var_local1};

                measurement meas{local, var};

                candidates.push_back({csv_meas.geometry_id, meas});
            }

            track_candidates.push_back(std::move(seed_params),
                                       std::move(candidates));
        }

        return track_candidates;
    }

    bound_track_parameters find_truth_param(const geometry_id surface_link,
                                            const measurement& meas) const {

        // Find the corresponding measurement
        auto is_same_meas = [&](const csv_measurement& csv_meas) {
            if (csv_meas.geometry_id == surface_link &&
                csv_meas.local0 == meas.local[0] &&
                csv_meas.local1 == meas.local[1]) {
                return true;
            }
            return false;
        };

        auto it = std::find_if(m_measurements.begin(), m_measurements.end(),
                               is_same_meas);
        assert(it != m_measurements.end());

        // Measurement index
        const auto m_id = std::distance(m_measurements.begin(), it);

        // Hit index
        const auto h_id = m_meas_hit_ids[m_id].hit_id;

        // Hit objects
        const auto hit = m_hits[h_id];

        // Particle index
        const auto p_id = hit.particle_id;

        // Particle obejcts
        const auto ptc = m_particles[p_id];

        // Get truth local position
        const point3 global_pos{hit.tx, hit.ty, hit.tz};
        const vector3 global_mom{hit.tpx, hit.tpy, hit.tpz};
        const auto truth_local = m_detector.get()->global_to_local(
            hit.geometry_id, global_pos, vector::normalize(global_mom));

        // Return value
        bound_track_parameters ret;
        auto& ret_vec = ret.vector();
        getter::element(ret_vec, e_bound_loc0, 0) = truth_local[0];
        getter::element(ret_vec, e_bound_loc1, 0) = truth_local[1];
        getter::element(ret_vec, e_bound_phi, 0) = it->phi;
        getter::element(ret_vec, e_bound_theta, 0) = it->theta;
        getter::element(ret_vec, e_bound_time, 0) = it->time;
        getter::element(ret_vec, e_bound_qoverp, 0) =
            ptc.q / std::sqrt(hit.tpx * hit.tpx + hit.tpy * hit.tpy +
                              hit.tpz * hit.tpz);
        return ret;
    }

    private:
    std::unique_ptr<detector_type> m_detector;
    std::vector<csv_meas_hit_id> m_meas_hit_ids;
    std::vector<csv_particle> m_particles;
    std::vector<csv_fatras_hit> m_hits;
    std::vector<csv_measurement> m_measurements;
};

}  // namespace traccc