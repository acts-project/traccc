/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <edm/internal_spacepoint.hpp>
#include <seeding/detail/doublet.hpp>
#include <seeding/detail/singlet.hpp>
#include <seeding/doublet_finding_helper.hpp>

namespace traccc {

/// Doublet finding to search the combinations of two compatible spacepoints
struct doublet_finding {
    /// Constructor for the doublet finding
    ///
    /// @param seedfinder_config is the configuration parameters
    /// @param isp_container is the internal spacepoint container
    doublet_finding(seedfinder_config& config) : m_config(config) {}

    /// Callable operator for doublet finding per middle spacepoint
    ///
    /// @param bin_information is the information of current bin
    /// @param spM_location is the location of the current middle spacepoint in
    /// internal spacepoint container
    /// @param bottom is whether it is for bottom or top spacepoints
    ///
    /// @return a pair of vectors of doublets and transformed coordinates
    std::pair<host_doublet_collection, host_lin_circle_collection> operator()(
        const host_internal_spacepoint_container& isp_container,
        const bin_information& bin_information, const sp_location& spM_location,
        bool bottom) {
        host_doublet_collection doublets;
        host_lin_circle_collection lin_circles;

        this->operator()(isp_container, bin_information, spM_location, doublets,
                         lin_circles, bottom);
        return std::make_pair(doublets, lin_circles);
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
    void operator()(const host_internal_spacepoint_container& isp_container,
                    const bin_information& bin_information,
                    const sp_location& spM_location,
                    host_doublet_collection& doublets,
                    host_lin_circle_collection& lin_circles, bool bottom) {
        const auto& spM =
            isp_container.items[spM_location.bin_idx][spM_location.sp_idx];
        const auto& rM = spM.radius();
        const auto& zM = spM.z();
        const auto& varianceRM = spM.varianceR();
        const auto& varianceZM = spM.varianceZ();

        auto& counts = bin_information.bottom_idx.counts;
        auto& bottom_bin_indices = bin_information.bottom_idx.vector_indices;

        // for middle-bottom doublets
        if (bottom) {
            for (size_t i = 0; i < counts; ++i) {
                auto& bin_idx = bottom_bin_indices[i];

                auto& spacepoints = isp_container.items[bin_idx];

                for (size_t sp_idx = 0; sp_idx < spacepoints.size(); ++sp_idx) {
                    auto& spB = spacepoints[sp_idx];

                    if (!doublet_finding_helper::isCompatible(
                            spM, spB, m_config, bottom)) {
                        continue;
                    }

                    lin_circle lin =
                        doublet_finding_helper::transform_coordinates(spM, spB,
                                                                      bottom);
                    sp_location spB_location = {bin_idx, sp_idx};
                    doublets.push_back(doublet({spM_location, spB_location}));
                    lin_circles.push_back(std::move(lin));
                }
            }
        }

        // for middle-top doublets
        else if (!bottom) {
            auto& counts = bin_information.top_idx.counts;
            auto& top_bin_indices = bin_information.top_idx.vector_indices;

            for (size_t i = 0; i < counts; ++i) {
                auto& bin_idx = top_bin_indices[i];
                auto& spacepoints = isp_container.items[bin_idx];

                for (size_t sp_idx = 0; sp_idx < spacepoints.size(); ++sp_idx) {
                    auto& spT = spacepoints[sp_idx];

                    if (!doublet_finding_helper::isCompatible(
                            spM, spT, m_config, bottom)) {
                        continue;
                    }

                    lin_circle lin =
                        doublet_finding_helper::transform_coordinates(spM, spT,
                                                                      bottom);
                    sp_location spT_location = {bin_idx, sp_idx};
                    doublets.push_back(doublet({spM_location, spT_location}));
                    lin_circles.push_back(std::move(lin));
                }
            }
        }
    }

    private:
    vecmem::memory_resource* m_resource;
    seedfinder_config m_config;
};

}  // namespace traccc
