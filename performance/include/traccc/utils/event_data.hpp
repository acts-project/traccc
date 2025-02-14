/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/particle.hpp"
#include "traccc/edm/silicon_cell_collection.hpp"
#include "traccc/edm/silicon_cluster_collection.hpp"
#include "traccc/edm/track_candidate.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/geometry/silicon_detector_description.hpp"
#include "traccc/io/csv/cell.hpp"
#include "traccc/io/csv/hit.hpp"
#include "traccc/io/csv/measurement.hpp"
#include "traccc/io/csv/measurement_hit_id.hpp"
#include "traccc/io/csv/particle.hpp"
#include "traccc/io/data_format.hpp"
#include "traccc/utils/seed_generator.hpp"

// Detray include(s).
#include <detray/io/frontend/detector_reader.hpp>

// Vecmem include(s).
#include <vecmem/memory/memory_resource.hpp>

// System include(s).
#include <map>
#include <string>

namespace traccc {

struct event_data {

    public:
    // Type definitions
    using detector_type = traccc::default_detector::host;

    event_data() = delete;

    /// Event data constructor
    ///
    /// @param[in] event_dir Event data directory
    /// data
    /// @param[in] event_id  Event id
    /// @param[in] use_acts_geom_source  Use acts geometry source
    /// @param[in] det       detray detector
    /// @param[in] format    file format
    /// @param[in] include_silicon_cells Use silicon cell data in object
    /// construction
    ///
    event_data(const std::string& event_dir, const std::size_t event_id,
               vecmem::memory_resource& resource,
               bool use_acts_geom_source = false,
               const detector_type* det = nullptr,
               data_format format = data_format::csv,
               bool include_silicon_cells = false);

    /// Fill the member variables related to CCA
    ///
    /// @param[in] cells cell EDM
    /// @param[in] cca_clusters cluster EDM from CCL algorithm
    /// @param[in] cca_measurements measurement EDM from measurement creation
    /// @param[in] dd    Detector description
    ///
    void fill_cca_result(
        const edm::silicon_cell_collection::host& cells,
        const edm::silicon_cluster_collection::host& cca_clusters,
        const measurement_collection_types::host& cca_measurements,
        const silicon_detector_description::host& dd);

    /// Generate truth candidate used for truth fitting
    ///
    /// @param[in] sg Seed generator for fitting
    /// @param[in] resource vecmem memory resource
    ///
    track_candidate_container_types::host generate_truth_candidates(
        seed_generator<detector_type>& sg, vecmem::memory_resource& resource);

    // Measurement map
    std::map<measurement_id, measurement> m_measurement_map;
    // Particle map
    std::map<particle_id, particle> m_particle_map;
    // Measurement to the contributing particle map
    std::map<measurement, std::map<particle, std::size_t>> m_meas_to_ptc_map;
    // CCA measurement to the contributing particle map
    std::map<measurement, std::map<particle, std::size_t>>
        m_found_meas_to_ptc_map;
    // Particle to its Measurements map
    std::map<particle, std::vector<measurement>> m_ptc_to_meas_map;
    // Measurement to its track parameter map
    std::map<measurement, std::pair<point3, point3>> m_meas_to_param_map;
    // CCA measurement to its track parameter map
    std::map<measurement, std::pair<point3, point3>> m_found_meas_to_param_map;
    // Cell to particle map
    std::map<io::csv::cell, particle> m_cell_to_particle_map;

    // Input arguments
    const std::string m_event_dir;
    const std::size_t m_event_id;
    std::reference_wrapper<vecmem::memory_resource> m_mr;

    private:
    void setup_csv(bool use_acts_geom_source, const detector_type* det,
                   bool include_silicon_cells);
};

}  // namespace traccc
