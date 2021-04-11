/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "definitions/algebra.hpp"
#include "definitions/primitives.hpp"
#include "edm/cluster.hpp"
#include "cpu/include/edm/cell_container.hpp"
#include "cpu/include/edm/measurement_container.hpp"
#include "definitions/algebra.hpp"

namespace traccc
{

    /// Connected component labeling.
    struct measurement_creation
    {

        /// Callable operator for the connected component, based on one single module
        ///
        /// @param clusters are the input cells into the connected component, they are
        ///              per module and unordered
        ///
        /// C++20 piping interface
        ///
        /// @return a measurement collection - usually same size or sometime slightly smaller than the input
        measurement_collection operator()(const cluster_collection &clusters) const
        {

            measurement_collection measurements;
            measurements.modcfg.placement = clusters.placement;
            this->operator()(clusters, measurements);
            return measurements;
        }

        /// Callable operator for the connected component, based on one single module
        ///
        /// @param clusters are the input cells into the connected component, they are
        ///              per module and unordered
        ///
        /// void interface
        ///
        /// @return a measurement collection - usually same size or sometime slightly smaller than the input
        void operator()(const cluster_collection &clusters, measurement_collection &measurements) const
        {
            // Assign the module id
            measurements.modcfg.module = clusters.module;
            // Run the algorithm
            measurements.items.reserve(clusters.items.size());
            for (const auto &cluster : clusters.items)
            {
                point2 p = {0., 0.};
                scalar totalWeight = 0.;

                // Should not happen
                if (cluster.cells.empty()){
                    continue;
                } 

                for (const auto &cell : cluster.cells)
                {
                    scalar weight = clusters.signal(cell.activation);
                    if (weight > clusters.threshold)
                    {
                        totalWeight += cell.activation;
                        auto cell_position = clusters.position_from_cell(cell.channel0, cell.channel1);
                        p = p + weight * cell_position;
                    }
                }
                if (totalWeight > 0.)
                {
                    measurement m;
                    m.local = 1. / totalWeight * p;
                    // @todo add variance estimation
                    measurements.items.push_back(std::move(m));
                }
            }
        }
    };

} // namespace traccc
