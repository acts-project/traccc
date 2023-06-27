/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <iostream>

#include "traccc/edm/container.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/spacepoint.hpp"
namespace traccc {

/// Seed consisting of three spacepoints, z origin and weight
struct seed {

    using link_type = spacepoint_collection_types::host::size_type;

    link_type spB_link;
    link_type spM_link;
    link_type spT_link;

    scalar weight;
    scalar z_vertex;

    TRACCC_HOST_DEVICE
    std::array<measurement_link, 3> get_measurements(
        const spacepoint_collection_types::const_view& spacepoints_view,
        const cell_module_collection_types::host& modules) const {
        const spacepoint_collection_types::const_device spacepoints(
            spacepoints_view);
        alt_measurement alt_measB = spacepoints.at(spB_link).meas;
        alt_measurement alt_measM = spacepoints.at(spM_link).meas;
        alt_measurement alt_measT = spacepoints.at(spT_link).meas;

        measurement measB = {alt_measB.local, alt_measB.variance};
        measurement measM = {alt_measM.local, alt_measM.variance};
        measurement measT = {alt_measT.local, alt_measT.variance};

        return {measurement_link{modules[alt_measB.module_link].module, measB},
                measurement_link{modules[alt_measM.module_link].module, measM},
                measurement_link{modules[alt_measT.module_link].module, measT}};
    }

    TRACCC_HOST_DEVICE
    std::array<alt_measurement, 3> get_measurements(
        const spacepoint_collection_types::const_view& spacepoints_view) const {
        const spacepoint_collection_types::const_device spacepoints(
            spacepoints_view);
        return {spacepoints.at(spB_link).meas, spacepoints.at(spM_link).meas,
                spacepoints.at(spT_link).meas};
    }

    TRACCC_HOST_DEVICE
    std::array<spacepoint, 3> get_spacepoints(
        const spacepoint_collection_types::const_view& spacepoints_view) const {
        const spacepoint_collection_types::const_device spacepoints(
            spacepoints_view);
        return {spacepoints.at(spB_link), spacepoints.at(spM_link),
                spacepoints.at(spT_link)};
    }
};

/// Declare all seed collection types
using seed_collection_types = collection_types<seed>;

}  // namespace traccc
