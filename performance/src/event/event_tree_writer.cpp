/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/event/event_tree_writer.hpp"

#include "event_tree_tool.hpp"

// ROOT include(s).
#ifdef TRACCC_HAVE_ROOT
#include <TFile.h>
#endif  // TRACCC_HAVE_ROOT

namespace traccc {

namespace details {

struct event_tree_writer_data {

    /// Plot tool for event data
    event_tree_tool m_event_tree_tool;
    event_tree_tool::event_tree_cache m_event_tree_cache;

};  // struct event_tree_writer_data

}  // namespace details

event_tree_writer::event_tree_writer(const config& cfg)
    : m_cfg(cfg), m_data(std::make_unique<details::event_tree_writer_data>()) {

    m_data->m_event_tree_tool.setup(m_data->m_event_tree_cache);
}

event_tree_writer::~event_tree_writer() {}

void event_tree_writer::write(
    const track_state_container_types::const_view& track_states_view,
    const event_map2& evt_map) {

    for (auto const& [key, ptc] : evt_map.ptc_map) {
        m_data->m_event_tree_tool.fill(m_data->m_event_tree_cache, ptc);
    }

    // Iterate over the track state container.
    track_state_container_types::const_device track_states(track_states_view);

    const unsigned int n_tracks = track_states.size();

    for (unsigned int i = 0; i < n_tracks; i++) {
        m_data->m_event_tree_tool.fill(m_data->m_event_tree_cache,
                                       track_states.at(i));
    }
}

void event_tree_writer::finalize() {

#ifdef TRACCC_HAVE_ROOT
    // Open the output file.
    std::unique_ptr<TFile> ofile(
        TFile::Open(m_cfg.file_path.c_str(), m_cfg.file_mode.c_str()));
    if ((!ofile) || ofile->IsZombie()) {
        throw std::runtime_error("Could not open output file \"" +
                                 m_cfg.file_path + "\" in mode \"" +
                                 m_cfg.file_mode + "\"");
    }
    ofile->cd();
#else
    std::cout << "ROOT file \"" << m_cfg.file_path << "\" is NOT created"
              << std::endl;
#endif  // TRACCC_HAVE_ROOT

    m_data->m_event_tree_tool.write(m_data->m_event_tree_cache);
}

}  // namespace traccc
