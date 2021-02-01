/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "edm/cell.hpp"
#include "edm/cluster.hpp"

namespace traccc {

    /// Connected component labelling.
    struct clustering {

        enum class connectivity : unsigned int {
            e_four = 4,
            e_eight = 8
        };

        connectivity _cell_connectivity = connectivity::e_four;

        /// Constructor with connectivity 
        clustering(connectivity cell_connectivity = connectivity::e_four)
         : _cell_connectivity(cell_connectivity)
        {}

        /// Callable operator for the connected component, based on one single module 
        ///
        /// @param cells are the input cells into the connected component, they are
        ///              per module and unordered
        /// @param opt the call options
        ///
        /// c++20 piping interface:
        /// @return a cluster collection  
        cluster_collection operator()(const cell_collection& cells) const {            
            cluster_collection clusters;
            this->operator()(cells, clusters);
            return clusters;
        }

        /// Callable operator for the connected component, based on one single module 
        ///
        /// @param cells are the input cells into the connected component, they are
        ///              per module and unordered
        /// @param clusters[in,out] are the output clusters 
        /// @param opt the call options
        ///
        /// void interface
        void operator()(const cell_collection& cells, cluster_collection& clusters) const {           
        }

    };

}
