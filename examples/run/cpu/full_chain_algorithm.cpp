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
    const silicon_detector_description::host& det_descr,
    detector_type* detector, std::unique_ptr<const traccc::Logger> logger)
    : messaging(logger->clone()),
      m_field_vec{0.f, 0.f, finder_config.bFieldInZ},
      m_field(detray::bfield::create_const_field<
              typename detector_type::scalar_type>(m_field_vec)),
      m_det_descr(det_descr),
      m_detector(detector),
      m_clusterization(mr, logger->cloneWithSuffix("ClusteringAlg")),
      m_spacepoint_formation(mr, logger->cloneWithSuffix("SpFormationAlg")),
      m_seeding(finder_config, grid_config, filter_config, mr,
                logger->cloneWithSuffix("SeedingAlg")),
      m_track_parameter_estimation(mr,
                                   logger->cloneWithSuffix("TrackParamEstAlg")),
      m_finding(finding_config, logger->cloneWithSuffix("TrackFindingAlg")),
      m_fitting(fitting_config, mr, logger->cloneWithSuffix("TrackFittingAlg")),
      m_finder_config(finder_config),
      m_grid_config(grid_config),
      m_filter_config(filter_config),
      m_finding_config(finding_config),
      m_fitting_config(fitting_config) {}

full_chain_algorithm::output_type full_chain_algorithm::operator()(
    const edm::silicon_cell_collection::host& cells) const {

    // Create a data object for the detector description.
    const silicon_detector_description::const_data det_descr_data =
        vecmem::get_data(m_det_descr.get());

    // Run the clusterization.
    auto cells_data = vecmem::get_data(cells);
    const clustering_algorithm::output_type measurements =
        m_clusterization(cells_data, det_descr_data);

    // If we have a Detray detector, run the seeding track finding and fitting.
    if (m_detector != nullptr) {

        // Run the seed-finding.
        const measurement_collection_types::const_view measurements_view =
            vecmem::get_data(measurements);
        const spacepoint_formation_algorithm::output_type spacepoints =
            m_spacepoint_formation(*m_detector, measurements_view);
        const edm::spacepoint_collection::const_data spacepoints_data =
            vecmem::get_data(spacepoints);
        const host::seeding_algorithm::output_type seeds =
            m_seeding(spacepoints_data);
        const edm::seed_collection::const_data seeds_data =
            vecmem::get_data(seeds);
        const host::track_params_estimation::output_type track_params =
            m_track_parameter_estimation(measurements_view, spacepoints_data,
                                         seeds_data, m_field_vec);
        const bound_track_parameters_collection_types::const_view
            track_params_view = vecmem::get_data(track_params);

        // Run the track finding.
        const finding_algorithm::output_type track_candidates = m_finding(
            *m_detector, m_field, measurements_view, track_params_view);

        // Run the track fitting, and return its results.
        const auto track_candidates_data = get_data(track_candidates);
        return m_fitting(*m_detector, m_field, track_candidates_data);
    }
    // If not, just return an empty object.
    else {

        // Return an empty object.
        return {};
    }
}

}  // namespace traccc
