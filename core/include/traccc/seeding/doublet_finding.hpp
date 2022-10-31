/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/edm/internal_spacepoint.hpp"
#include "traccc/seeding/detail/doublet.hpp"
#include "traccc/seeding/detail/singlet.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"
#include "traccc/seeding/detail/spacepoint_type.hpp"
#include "traccc/seeding/doublet_finding_helper.hpp"
#include "traccc/utils/algorithm.hpp"

namespace traccc {

/// Doublet finding to search the combinations of two compatible spacepoints
/// @tparam otherSpType is whether it is for middle-bottom or middle-top doublet
template <details::spacepoint_type otherSpType>
struct doublet_finding
    : public algorithm<std::pair<doublet_collection_types::host,
                                 lin_circle_collection_types::host>(
          const sp_grid&, const sp_location&)> {

    static_assert(otherSpType == details::spacepoint_type::bottom ||
                  otherSpType == details::spacepoint_type::top);

    /// Constructor for the doublet finding
    ///
    /// @param seedfinder_config is the configuration parameters
    /// @param isp_container is the internal spacepoint container
    doublet_finding(const seedfinder_config& config) : m_config(config) {}

    /// Callable operator for doublet finding per middle spacepoint
    ///
    /// @param bin_information is the information of current bin
    /// @param spM_location is the location of the current middle spacepoint in
    /// internal spacepoint container
    ///
    /// @return a pair of vectors of doublets and transformed coordinates
    output_type operator()(const sp_grid& g2,
                           const sp_location& l) const override {
        output_type result;
        this->operator()(g2, l, result);
        return result;
    }

    /// Callable operator for doublet finding of a middle spacepoint
    ///
    /// @param bin_information is the information of current bin
    /// @param spM_location is the location of the current middle spacepoint in
    /// internal spacepoint container
    ///
    /// void interface
    ///
    /// @return a pair of vectors of doublets and transformed coordinates
    void operator()(const sp_grid& g2, const sp_location& l,
                    output_type& o) const {
        // output
        auto& doublets = o.first;
        auto& lin_circles = o.second;

        // middle spacepoint
        const auto& spM = g2.bin(l.bin_idx)[l.sp_idx];

        auto phi_bins = g2.axis_p0().zone(spM.phi(), m_config.neighbor_scope);
        auto z_bins = g2.axis_p1().zone(spM.z(), m_config.neighbor_scope);

        // iterator over neighbor bins
        for (auto& phi_bin : phi_bins) {
            for (auto& z_bin : z_bins) {
                auto bin_idx = phi_bin + z_bin * g2.axis_p0().bins();

                const auto& neighbors = g2.bin(phi_bin, z_bin);
                for (unsigned int sp_idx = 0; sp_idx < neighbors.size();
                     sp_idx++) {
                    const auto& sp_nb = neighbors[sp_idx];

                    if (!doublet_finding_helper::isCompatible<otherSpType>(
                            spM, sp_nb, m_config)) {
                        continue;
                    }

                    lin_circle lin =
                        doublet_finding_helper::transform_coordinates<
                            otherSpType>(spM, sp_nb);
                    sp_location sp_nb_location = {
                        static_cast<unsigned int>(bin_idx),
                        static_cast<unsigned int>(sp_idx)};
                    doublets.push_back(doublet({l, sp_nb_location}));
                    lin_circles.push_back(std::move(lin));
                }
            }
        }
    }

    private:
    seedfinder_config m_config;
};

}  // namespace traccc
