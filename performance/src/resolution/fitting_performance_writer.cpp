/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/resolution/fitting_performance_writer.hpp"

namespace traccc {

fitting_performance_writer::fitting_performance_writer(
    fitting_performance_writer::config cfg)
    : m_cfg(std::move(cfg)), m_res_plot_tool(m_cfg.res_plot_tool_config) {

    const auto& path = m_cfg.file_path;

    m_output_file = std::unique_ptr<TFile>{
        TFile::Open(path.c_str(), m_cfg.file_mode.c_str())};

    if (not m_output_file) {
        throw std::invalid_argument("Could not open '" + path + "'");
    }

    m_res_plot_tool.book(m_res_plot_cache);
}

fitting_performance_writer::~fitting_performance_writer() {

    m_res_plot_tool.clear(m_res_plot_cache);

    if (m_output_file) {
        m_output_file->Close();
    }
}

void fitting_performance_writer::finalize() {
    m_output_file->cd();
    m_res_plot_tool.write(m_res_plot_cache);
}

}  // namespace traccc
