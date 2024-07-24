/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/edm/cell.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/utils/algorithm.hpp"
#include "vecmem/memory/memory_resource.hpp"

namespace traccc::futhark {
struct component_connection : algorithm<measurement_container_types::host(
                                  const cell_container_types::host& cells)> {
    component_connection(vecmem::memory_resource&);

    output_type operator()(const cell_container_types::host& cells) const;

    private:
    vecmem::memory_resource& m_mr;
};

}  // namespace traccc::futhark
