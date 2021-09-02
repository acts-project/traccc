/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <algorithm>

namespace traccc {

/// class (experiment-dependent) for estimating the number of multiples
/// as a function of number of spacepoints in the spacepoint bin
class multiplet_estimator {
    public:
    multiplet_estimator() = default;
    ~multiplet_estimator() = default;

    struct pol1_params {
        scalar p0;
        scalar p1;

        size_t operator()(size_t n) const { return size_t(p0 + p1 * n); }
    };

    struct pol2_params {
        scalar p0;
        scalar p1;
        scalar p2;

        size_t operator()(size_t n) const {
            return size_t(p0 + p1 * n + p2 * n * n);
        }
    };

    struct config {
        scalar safety_factor = 2.0;
        scalar safety_adder = 10;
        // mid-bot doublets size allocation parameter
        pol2_params par_for_mb_doublets = {1, 28.77, 0.4221};
        // mid-top doublets size allocation parameter
        pol2_params par_for_mt_doublets = {1, 19.73, 0.232};
        // triplets size allocation parameter
        pol2_params par_for_triplets = {1, 0, 0.02149};
        // seeds size allocation parameter
        pol1_params par_for_seeds = {0, 0.3431};
    };

    /// Estimate the number of mid-bot doublets in the bin
    ///
    /// @param n_spm the number of middles spacepoints in the bin
    /// @return the number of mid-bot doublets
    size_t get_mid_bot_doublets_size(int n_spM) const {
        return (m_cfg.par_for_mb_doublets(n_spM) + m_cfg.safety_adder) *
               m_cfg.safety_factor;
    }

    /// Estimate the number of mid-top doublets in the bin
    ///
    /// @param n_spm the number of middles spacepoints in the bin
    /// @return the number of mid-top doublets
    size_t get_mid_top_doublets_size(int n_spM) const {
        return (m_cfg.par_for_mt_doublets(n_spM) + m_cfg.safety_adder) *
               m_cfg.safety_factor;
    }

    /// Estimate the number of triplets in the bin
    ///
    /// @param n_spm the number of middles spacepoints in the bin
    /// @return the number of triplets
    size_t get_triplets_size(int n_spM) const {
        return (m_cfg.par_for_triplets(n_spM) + m_cfg.safety_adder) *
               m_cfg.safety_factor;
    }

    /// Estimate the number of seeds in the event
    ///
    /// @param n_internal_sp the number of internal spacepoints in the event
    /// @return the number of seeds
    size_t get_seeds_size(int n_internal_sp) const {
        return (m_cfg.par_for_seeds(n_internal_sp) + m_cfg.safety_adder) *
               m_cfg.safety_factor;
    }

    config m_cfg;

    private:
};

}  // namespace traccc
