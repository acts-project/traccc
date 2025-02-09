/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/resolution/stat_plot_tool_config.hpp"
#include "traccc/utils/helpers.hpp"

// Project include(s).
#include "traccc/edm/track_candidate.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/utils/event_data.hpp"

// System include(s).
#include <map>
#include <memory>
#include <string>
#include <string_view>

namespace traccc {
namespace details {

/// Data members that should not pollute the API of
/// @c traccc::finding_performance_writer
struct finding_performance_writer_data;

}  // namespace details

class finding_performance_writer {

    public:
    /// Configuration for the tool
    struct config {

        // Algorithm name, for ROOT display
        std::string algorithm_name = "finding";
        /// Output filename.
        std::string file_path = "performance_track_finding.root";
        /// Output file mode
        std::string file_mode = "RECREATE";

        /// Plot tool configurations.
        std::map<std::string, plot_helpers::binning> var_binning = {
            {"Eta", plot_helpers::binning("#eta", 40, -4.f, 4.f)},
            {"Phi", plot_helpers::binning("#phi", 100, -3.15f, 3.15f)},
            {"Pt", plot_helpers::binning("p_{T} [GeV/c]", 40, 0.f, 100.f)},
            {"Num", plot_helpers::binning("N", 30, -0.5f, 29.5f)}};

        /// Cut values
        scalar pT_cut = 0.5f * traccc::unit<scalar>::GeV;
        scalar z_min = -500.f * traccc::unit<scalar>::mm;
        scalar z_max = 500.f * traccc::unit<scalar>::mm;
        scalar r_max = 200.f * traccc::unit<scalar>::mm;
        scalar matching_ratio = 0.5f;
        bool double_matching = true;

        stat_plot_tool_config stat_config;
    };

    /// Construct from configuration and log level.
    /// @param cfg The configuration
    ///
    finding_performance_writer(const config& cfg);

    /// Destructor
    ~finding_performance_writer();

    void write(const track_candidate_container_types::const_view&
                   track_candidates_view,
               const event_data& evt_data);

    void write(const track_state_container_types::const_view& track_states_view,
               const event_data& evt_data);

    void finalize();

    private:
    /// Configuration for the tool
    config m_cfg;

    /// Opaque data members for the class
    std::unique_ptr<details::finding_performance_writer_data> m_data;

    /// Common method to both track finding and ambiguity resolution
    void write_common(const std::vector<std::vector<measurement>>& tracks,
                      const event_data& evt_data);

};  // class finding_performance_writer

}  // namespace traccc
