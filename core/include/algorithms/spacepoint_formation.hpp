/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "edm/measurement.hpp"
#include "edm/spacepoint.hpp"

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
            spacepoints.items.reserve(measurements.items.size());
        }

    };
    
}