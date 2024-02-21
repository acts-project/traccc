/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/utils/helpers.hpp"

// Project include(s).
#include "traccc/edm/track_state.hpp"
#include "traccc/io/event_map2.hpp"

// System include(s).
#include <map>
#include <memory>
#include <string>
#include <string_view>

// Adapted from finding_performance_writer.hpp

namespace traccc {
namespace details {

/// Data members that should not pollute the API of
/// @c traccc::ambiguity_resolution_performance_writer
struct ambiguity_resolution_performance_writer_data;

}  // namespace details

class ambiguity_resolution_performance_writer {

    public:
    /// Configuration for the tool
    struct config {

        /// Output filename.
        std::string file_path = "performance_track_ambiguity_resolution.root";
        /// Output file mode
        std::string file_mode = "RECREATE";

        /// Plot tool configurations.
        std::map<std::string, plot_helpers::binning> var_binning = {
            {"Eta", plot_helpers::binning("#eta", 40, -4, 4)},
            {"Phi", plot_helpers::binning("#phi", 100, -3.15, 3.15)},
            {"Pt", plot_helpers::binning("p_{T} [GeV/c]", 40, 0, 100)},
            {"Num", plot_helpers::binning("N", 30, -0.5, 29.5)}};

        /// Cut values
        scalar pT_cut = 0.1f * traccc::unit<scalar>::GeV;
    };

    /// Construct from configuration and log level.
    /// @param cfg The configuration
    ///
    ambiguity_resolution_performance_writer(const config& cfg);

    /// Destructor
    ~ambiguity_resolution_performance_writer();

    void write(const typename track_state_container_types::const_view&
                   track_states_view,
               const event_map2& evt_map);

    void finalize();

    private:
    /// Configuration for the tool
    config m_cfg;

    /// Opaque data members for the class
    std::unique_ptr<details::ambiguity_resolution_performance_writer_data>
        m_data;

};  // class ambiguity_resolution_performance_writer

}  // namespace traccc
