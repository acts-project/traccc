/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "cpu/include/edm/measurement_container.hpp"
#include "cpu/include/edm/spacepoint_container.hpp"

namespace traccc {

    /// Connected component labeling.
    struct spacepoint_formation {

        /// Callable operator for the space point formation, based on one single module 
        ///
        /// @param measurements are the input measurements, in this pixel demonstrator it one space
        ///    point per measurement
        ///
        /// C++20 piping interface
        ///
        /// @return a measurement collection - size of input/output container is identical
        spacepoint_collection operator()(const measurement_collection& measurements) const {
            
            spacepoint_collection spacepoints;
            this->operator()(measurements, spacepoints);
            return spacepoints;
        }

        /// Callable operator for the space point formation, based on one single module 
        ///
        /// @param measurements are the input measurements, in this pixel demonstrator it one space
        ///    point per measurement
        ///
        /// void interface
        ///
        /// @return a measurement collection - size of input/output container is identical
        void operator()(const measurement_collection& measurements, spacepoint_collection& spacepoints) const {
            // Assign the module id
            spacepoints.module = measurements.modcfg.module;
            // Run the algorithm
            spacepoints.items.reserve(measurements.items.size());
            for (const auto& m : measurements.items){
                spacepoint s;
                point3 local_3d = {m.local[0], m.local[1], 0.};
                s.global = measurements.modcfg.placement.point_to_global(local_3d);
                // @todo add variance estimation
                spacepoints.items.push_back(std::move(s));
            }
        }

    };
    
}
