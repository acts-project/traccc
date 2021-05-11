/*
 * TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "edm/cell.hpp"
#include "edm/measurement.hpp"

#include "cuda/algorithms/component_connection.hpp"

namespace traccc::cuda {
    host_measurement_collection
    component_connection::operator()(
        const host_cell_container & data
    ) const {
        return {};
    }
}
