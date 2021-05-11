/*
 * TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "edm/cell.hpp"
#include "edm/cluster.hpp"
#include "edm/measurement.hpp"

namespace traccc::cuda {
    struct component_connection {
        host_measurement_collection
        operator()(
            const host_cell_container & cells
        ) const;
    };
}
