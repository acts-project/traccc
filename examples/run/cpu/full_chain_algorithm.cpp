/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "full_chain_algorithm.hpp"

namespace traccc {

full_chain_algorithm::full_chain_algorithm(
    vecmem::memory_resource& mr, const clustering_algorithm::config_type&,
    const seedfinder_config& finder_config,
    const spacepoint_grid_config& grid_config,
    const seedfilter_config& filter_config,
    const finding_algorithm::config_type& finding_config,
    const fitting_algorithm::config_type& fitting_config,
    detector_type* detector)
    : m_field_vec{0.f, 0.f, finder_config.bFieldInZ},
      m_field(detray::bfield::create_const_field(m_field_vec)),
      m_detector(detector),
      m_clusterization(mr),
      m_spacepoint_formation(mr),
      m_seeding(finder_config, grid_config, filter_config, mr),
      m_track_parameter_estimation(mr),
      m_finding(finding_config),
      m_fitting(fitting_config),
      m_finder_config(finder_config),
      m_grid_config(grid_config),
      m_filter_config(filter_config),
      m_finding_config(finding_config),
      m_fitting_config(fitting_config) {}

full_chain_algorithm::output_type full_chain_algorithm::operator()(
    const cell_collection_types::host& cells,
    const cell_module_collection_types::host& modules) const {

    // Run the clusterization.
    const host::clusterization_algorithm::output_type measurements =
        m_clusterization(vecmem::get_data(cells), vecmem::get_data(modules));

    // Run the seed-finding.
    const host::spacepoint_formation_algorithm::output_type spacepoints =
        m_spacepoint_formation(vecmem::get_data(measurements),
                               vecmem::get_data(modules));
    const track_params_estimation::output_type track_params =
        m_track_parameter_estimation(spacepoints, m_seeding(spacepoints),
                                     m_field_vec);

    // If we have a Detray detector, run the track finding and fitting.
    if (m_detector != nullptr) {

        // Return the final container, after track finding and fitting.
        return m_fitting(
            *m_detector, m_field,
            m_finding(*m_detector, m_field, measurements, track_params));

    }
    // If not, just return an empty object.
    else {

        // Return an empty object.
        return {};
    }
}

}  // namespace traccc
