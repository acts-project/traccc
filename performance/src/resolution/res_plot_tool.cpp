/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/resolution/res_plot_tool.hpp"

namespace traccc {

res_plot_tool::res_plot_tool(const config& cfg) : m_cfg(cfg) {}

void res_plot_tool::book(res_plot_tool::res_plot_cache& cache) const {

    plot_helpers::binning b_pull = m_cfg.var_binning.at("pull");

    for (std::size_t idx = 0; idx < e_bound_size; idx++) {
        std::string par_name = m_cfg.param_names.at(idx);

        // Binning for residual is parameter dependent
        std::string par_residual = "residual_" + par_name;
        plot_helpers::binning b_residual = m_cfg.var_binning.at(par_residual);

        // residual distributions
        cache.residuals[par_name] = plot_helpers::book_histo(
            Form("res_%s", par_name.c_str()),
            Form("Residual of %s", par_name.c_str()), b_residual);

        // pull distritutions
        cache.pulls[par_name] = plot_helpers::book_histo(
            Form("pull_%s", par_name.c_str()),
            Form("Pull of %s", par_name.c_str()), b_pull);
    }
}

void res_plot_tool::fill(res_plot_tool::res_plot_cache& cache,
                         const bound_track_parameters& truth_param,
                         const bound_track_parameters& fit_param) const {

    for (std::size_t idx = 0; idx < e_bound_size; idx++) {
        std::string par_name = m_cfg.param_names.at(idx);

        const auto residual = getter::element(fit_param.vector(), idx, 0) -
                              getter::element(truth_param.vector(), idx, 0);

        const auto pull = residual / std::sqrt(getter::element(
                                         fit_param.covariance(), idx, idx));

        //// DELETE ME ////
        /*
        if (idx == 3) {
            printf("fit %f  \n", getter::element(fit_param.vector(), idx, 0));
            printf("truth %f  \n",
                   getter::element(truth_param.vector(), idx, 0));
            printf("cov %.10lf  \n", std::sqrt(getter::element(
                                        fit_param.covariance(), idx, idx)));
            printf("res %f  \n", residual);
            printf("pull %f  \n", pull);
        }
        */

        plot_helpers::fill_histo(cache.residuals.at(par_name), residual);
        plot_helpers::fill_histo(cache.pulls.at(par_name), pull);
    }
}

void res_plot_tool::write(const res_plot_tool::res_plot_cache& cache) const {
    for (std::size_t idx = 0; idx < e_bound_size; idx++) {
        std::string par_name = m_cfg.param_names.at(idx);
        cache.residuals.at(par_name)->Write();
        cache.pulls.at(par_name)->Write();
    }
}

void res_plot_tool::clear(const res_plot_tool::res_plot_cache& cache) const {
    for (std::size_t idx = 0; idx < e_bound_size; idx++) {
        std::string par_name = m_cfg.param_names.at(idx);
        delete cache.residuals.at(par_name);
        delete cache.pulls.at(par_name);
    }
}

}  // namespace traccc