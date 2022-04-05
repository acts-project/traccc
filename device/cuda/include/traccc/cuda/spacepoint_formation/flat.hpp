/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/cuda/geometry/module_map.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/utils/algorithm.hpp"

namespace traccc::cuda {
/**
 * @brief CUDA implementation of spacepoint formation using a flat EDM.
 *
 * This is an implementation of spacepoint formation, which transforms
 * two-dimensional points on detector surfaces to three-dimensional points in
 * global space.
 *
 * This implementation uses a flat EDM, in which points are represented as a
 * product type of the detector ID and the measurement itself.
 */
struct spacepoint_formation_flat : public algorithm<host_spacepoint_collection(
                                       const host_measurement_container&)> {
    public:
    /**
     * @brief Construct a new spacepoint formation algorithm object
     *
     * @param mr The memory resource to use for the management of memory.
     * @param mm The module map to use for transformation lookups.
     */
    spacepoint_formation_flat(vecmem::memory_resource& mr,
                              const module_map<geometry_id, transform3>& mm);

    /**
     * @brief Execute spacepoint formation.
     *
     * @param m A container of measurements.
     * @return A collection of spacepoints.
     */
    host_spacepoint_collection operator()(
        const host_measurement_container& m) const override;

    private:
    /**
     * @brief The memory resource that is used to manage all allocations and
     * deallocations that this algorithm performs.
     */
    vecmem::memory_resource& m_mr;

    /**
     * @brief The module map that is used to look up transformation matrices on
     * the CUDA device.
     */
    const module_map<geometry_id, transform3>& m_mm;
};
}  // namespace traccc::cuda
