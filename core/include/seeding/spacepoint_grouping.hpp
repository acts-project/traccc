/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <algorithm>
#include <edm/internal_spacepoint.hpp>
#include <edm/spacepoint.hpp>
#include <iostream>
#include <seeding/detail/bin_finder.hpp>
#include <seeding/detail/seeding_config.hpp>
#include <seeding/detail/spacepoint_grid.hpp>

namespace traccc {

// group the spacepoints basaed on its position
struct spacepoint_grouping {
    // constructor declaration
    spacepoint_grouping(const seedfinder_config& config,
                        const spacepoint_grid_config& grid_config,
                        vecmem::memory_resource* mr);

    host_internal_spacepoint_container operator()(
        const host_spacepoint_container& sp_container) {
        host_internal_spacepoint_container internal_sp_container(m_mr);
        this->operator()(sp_container, internal_sp_container);

        return internal_sp_container;
    }

    std::shared_ptr<spacepoint_grid> get_spgrid() { return m_spgrid; }

    void operator()(const host_spacepoint_container& sp_container,
                    host_internal_spacepoint_container& internal_sp_container) {
        // get region of interest (or full detector if configured accordingly)
        scalar phiMin = m_config.phiMin;
        scalar phiMax = m_config.phiMax;
        scalar zMin = m_config.zMin;
        scalar zMax = m_config.zMax;

        // sort by radius
        // add magnitude of beamPos to rMax to avoid excluding measurements
        // create number of bins equal to number of millimeters rMax
        // (worst case minR: m_configured minR + 1mm)

        size_t numRBins = (m_config.rMax + getter::norm(vector2{m_config.beamPos_x, m_config.beamPos_y}));
        std::vector<std::vector<internal_spacepoint<spacepoint>>> rBins(
            numRBins);

        for (auto& sp_vec : sp_container.get_items()) {
            for (auto& sp : sp_vec) {
                scalar spX = sp.global[0];
                scalar spY = sp.global[1];
                scalar spZ = sp.global[2];
                scalar varR = sp.variance[0];  // Need a check
                scalar varZ = sp.variance[1];

                if (spZ > zMax || spZ < zMin) {
                    continue;
                }
                scalar spPhi = std::atan2(spY, spX);
                if (spPhi > phiMax || spPhi < phiMin) {
                    continue;
                }

                // Note: skip covTool of ACTS main repository
                // vector2 variance = covTool(sp, m_config.zAlign,
                // m_config.rAlign, m_config.sigmaError);
                vector2 variance({varR, varZ});
                vector3 spPosition({spX, spY, spZ});

                auto isp = internal_spacepoint<spacepoint>(
                    sp, spPosition, vector2{m_config.beamPos_x, m_config.beamPos_y}, variance);
                // calculate r-Bin index and protect against overflow (underflow
                // not possible)
                size_t rIndex = isp.radius();
                // if index out of bounds, the SP is outside the region of
                // interest
                if (rIndex >= numRBins) {
                    continue;
                }
                rBins[rIndex].push_back(std::move(isp));
            }
        }

        // fill rbins into grid such that each grid bin is sorted in r
        // space points with delta r < rbin size can be out of order
        // iterate over all bins (without under/overflow bins)
        auto local_bins = m_spgrid->numLocalBins();
        for (size_t i = 1; i <= local_bins[0]; ++i) {
            for (size_t j = 1; j <= local_bins[1]; ++j) {
                std::array<size_t, 2> local_bin({i, j});

                size_t global_bin = m_spgrid->globalBinFromLocalBins(local_bin);
                auto bottom_indices = m_bottom_bin_finder->find_bins(
                    local_bin[0], local_bin[1], m_spgrid.get());
                auto top_indices = m_top_bin_finder->find_bins(
                    local_bin[0], local_bin[1], m_spgrid.get());

                bin_information bin_info;
                bin_info.global_index = global_bin;
                bin_info.bottom_idx.counts = bottom_indices.size();
                bin_info.top_idx.counts = top_indices.size();

                std::copy(bottom_indices.begin(), bottom_indices.end(),
                          &bin_info.bottom_idx.global_indices[0]);

                std::copy(top_indices.begin(), top_indices.end(),
                          &bin_info.top_idx.global_indices[0]);

                internal_sp_container.push_back(
                    std::move(bin_info),
                    vecmem::vector<internal_spacepoint<spacepoint>>(m_mr));
            }
        }

        for (auto& rbin : rBins) {
            for (auto& isp : rbin) {
                vector2 spLocation({isp.phi(), isp.z()});
                auto local_bin = m_spgrid->localBinsFromPosition(spLocation);
                auto global_bin = m_spgrid->globalBinFromLocalBins(local_bin);

                auto location = find_vector_id_from_global_id(
                    global_bin, internal_sp_container.get_headers());
                internal_sp_container.at(location).items.push_back(
                    std::move(isp));
            }
        }

        fill_vector_id(internal_sp_container);
    }

    private:
    seedfinder_config m_config;
    spacepoint_grid_config m_grid_config;
    std::shared_ptr<spacepoint_grid> m_spgrid;
    std::unique_ptr<bin_finder> m_bottom_bin_finder;
    std::unique_ptr<bin_finder> m_top_bin_finder;
    vecmem::memory_resource* m_mr;
};

spacepoint_grouping::spacepoint_grouping(
    const seedfinder_config& config, const spacepoint_grid_config& grid_config,
    vecmem::memory_resource* mr)
    : m_config(config), m_grid_config(grid_config), m_mr(mr) {
    // calculate circle intersections of helix and max detector radius
    scalar minHelixRadius =
        grid_config.minPt / (300. * grid_config.bFieldInZ);  // in mm
    scalar maxR2 = grid_config.rMax * grid_config.rMax;
    scalar xOuter = maxR2 / (2 * minHelixRadius);
    scalar yOuter = std::sqrt(maxR2 - xOuter * xOuter);
    scalar outerAngle = std::atan(xOuter / yOuter);
    // intersection of helix and max detector radius minus maximum R distance
    // from middle SP to top SP
    scalar innerAngle = 0;
    if (grid_config.rMax > grid_config.deltaRMax) {
        scalar innerCircleR2 = (grid_config.rMax - grid_config.deltaRMax) *
                               (grid_config.rMax - grid_config.deltaRMax);
        scalar xInner = innerCircleR2 / (2 * minHelixRadius);
        scalar yInner = std::sqrt(innerCircleR2 - xInner * xInner);
        innerAngle = std::atan(xInner / yInner);
    }

    // FIXME: phibin size must include max impact parameters
    // divide 2pi by angle delta to get number of phi-bins
    // size is always 2pi even for regions of interest
    int phiBins = std::floor(2 * M_PI / (outerAngle - innerAngle));
    Acts::detail::Axis<Acts::detail::AxisType::Equidistant,
                       Acts::detail::AxisBoundaryType::Closed>
        phiAxis(-M_PI, M_PI, phiBins);

    // TODO: can probably be optimized using smaller z bins
    // and returning (multiple) neighbors only in one z-direction for forward
    // seeds
    // FIXME: zBinSize must include scattering

    scalar zBinSize = grid_config.cotThetaMax * grid_config.deltaRMax;
    int zBins = std::floor((grid_config.zMax - grid_config.zMin) / zBinSize);
    Acts::detail::Axis<Acts::detail::AxisType::Equidistant,
                       Acts::detail::AxisBoundaryType::Bound>
        zAxis(grid_config.zMin, grid_config.zMax, zBins);

    m_spgrid = std::make_shared<spacepoint_grid>(
        spacepoint_grid(std::make_tuple(phiAxis, zAxis)));
}
}  // namespace traccc
