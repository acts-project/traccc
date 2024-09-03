/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/utils/helpers.hpp"

// Project include(s).
#include "traccc/edm/track_state.hpp"
#include "traccc/io/event_map2.hpp"

namespace traccc {
namespace details {

/// Data members that should not pollute the API of
/// @c traccc::track_property_writer
struct track_property_writer_data;

}  // namespace details

class track_property_writer {

    public:
    /// Configuration for the tool
    struct config {

        /// Output filename.
        std::string file_path = "track_property.root";
        /// Output file mode
        std::string file_mode = "RECREATE";
    };

    /// Construct from configuration and log level.
    /// @param cfg The configuration
    ///
    track_property_writer(const config& cfg);

    /// Destructor
    ~track_property_writer();

    void write(const track_state_container_types::const_view& track_states_view,
               const event_map2& evt_map);

    void finalize();

    private:
    /// Configuration for the tool
    config m_cfg;

    /// Opaque data members for the class
    std::unique_ptr<details::track_property_writer_data> m_data;

};  // class track_property_writer

}  // namespace traccc