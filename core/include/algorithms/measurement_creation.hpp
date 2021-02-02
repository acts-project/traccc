/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "edm/cell.hpp"
#include "edm/cluster.hpp"
#include "edm/measurement.hpp"

namespace traccc {

    /// Connected component labeling.
    struct measurement_creation {

        /// Callable operator for the connected component, based on one single module 
        ///
        /// @param clusters are the input cells into the connected component, they are
        ///              per module and unordered
        ///
        /// C++20 piping interface
        ///
        /// @return a measurement collection - usually same size or sometime slightly smaller than the input
        measurement_collection operator()(const cluster_collection& clusters) const {
            
            measurement_collection measurements;
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
        void operator()(const cluster_collection& clusters, measurement_collection& measurements) const {
            measurements.items.reserve(clusters.items.size());

        }

    };
    
}