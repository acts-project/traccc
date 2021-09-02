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

#include "utils/algorithm.hpp"

namespace traccc {

/// Doublet finding to search the combinations of two compatible spacepoints
struct doublet_finding
    : public algorithm<
          std::tuple<const host_internal_spacepoint_container&,
                     const bin_information&, const sp_location&, const bool>,
          std::pair<host_doublet_collection, host_lin_circle_collection> > {

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
    /// @param bottom is whether it is for bottom or top spacepoints
    ///
    /// @return a pair of vectors of doublets and transformed coordinates
    output_type operator()(const input_type& i) const override {
        output_type result;
        this->operator()(i, result);
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
    void operator()(const input_type& i, output_type& o) const {

        // input
        const auto& isp_container = std::get<0>(i);
        const auto& bin_information = std::get<1>(i);
        const auto& spM_location = std::get<2>(i);
        const auto& bottom = std::get<3>(i);

        // output
        auto& doublets = o.first;
        auto& lin_circles = o.second;

        // Run the algorithm
        const auto& spM =
            isp_container
                .get_items()[spM_location.bin_idx][spM_location.sp_idx];

        auto& counts = bin_information.bottom_idx.counts;
        auto& bottom_bin_indices = bin_information.bottom_idx.vector_indices;

        // for middle-bottom doublets
        if (bottom) {
            for (size_t i = 0; i < counts; ++i) {
                auto& bin_idx = bottom_bin_indices[i];

                auto& spacepoints = isp_container.get_items()[bin_idx];

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
                auto& spacepoints = isp_container.get_items()[bin_idx];

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
