/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/resolution/fitting_performance_writer.hpp"

#include "res_plot_tool.hpp"

// ROOT include(s).
#ifdef TRACCC_HAVE_ROOT
#include <TFile.h>
#endif  // TRACCC_HAVE_ROOT

// System include(s).
#include <iostream>
#include <memory>
#include <stdexcept>

namespace traccc {
namespace details {

struct fitting_performance_writer_data {

    /// Constructor
    fitting_performance_writer_data(
        const fitting_performance_writer::config& cfg)
        : m_res_plot_tool(cfg.config) {}

    /// Plot tool for resolution
    res_plot_tool m_res_plot_tool;
    res_plot_tool::res_plot_cache m_res_plot_cache;
};

}  // namespace details

fitting_performance_writer::fitting_performance_writer(const config& cfg)
    : m_cfg(cfg),
      m_data(std::make_unique<details::fitting_performance_writer_data>(cfg)) {

    m_data->m_res_plot_tool.book(m_data->m_res_plot_cache);
}

fitting_performance_writer::~fitting_performance_writer() {}

void fitting_performance_writer::finalize() {

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

    m_data->m_res_plot_tool.write(m_data->m_res_plot_cache);
}

void fitting_performance_writer::write_impl(
    const bound_track_parameters& truth_param,
    const bound_track_parameters& fit_param) {

    m_data->m_res_plot_tool.fill(m_data->m_res_plot_cache, truth_param,
                                 fit_param);
}

}  // namespace traccc
