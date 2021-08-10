/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc {

/// virtual class (experiment-dependent) for estimating the number of multiples
/// as a function of number of spacepoints in the spacepoint bin
class stats_config {
    public:
    virtual ~stats_config() = default;

    /// Estimate the number of mid-bot doublets in the bin
    ///
    /// @param n_spm the number of middles spacepoints in the bin
    /// @return the number of mid-bot doublets
    virtual size_t get_mid_bot_doublets_size(int n_spM) const = 0;

    /// Estimate the number of mid-top doublets in the bin
    ///
    /// @param n_spm the number of middles spacepoints in the bin
    /// @return the number of mid-top doublets
    virtual size_t get_mid_top_doublets_size(int n_spM) const = 0;

    /// Estimate the number of triplets in the bin
    ///
    /// @param n_spm the number of middles spacepoints in the bin
    /// @return the number of triplets
    virtual size_t get_triplets_size(int n_spM) const = 0;

    /// Estimate the number of seeds in the event
    ///
    /// @param n_internal_sp the number of internal spacepoints in the event
    /// @return the number of seeds
    virtual size_t get_seeds_size(int n_internal_sp) const = 0;
};

}  // namespace traccc
