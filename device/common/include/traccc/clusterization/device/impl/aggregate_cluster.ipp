/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include <vecmem/memory/device_atomic_ref.hpp>

#include "traccc/clusterization/detail/measurement_creation_helper.hpp"

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void aggregate_cluster(
    const alt_cell_collection_types::const_device& cells,
    const cell_module_collection_types::const_device& modules,
    const unsigned short* f, const ccl_partition& part,
    const unsigned short tid, alt_measurement* out, unsigned int& outi) {

    /*
     * This is the post-processing stage, where we merge the clusters into a
     * single measurement and write it to the output.
     */

    /*
     * If and only if the value in the work arrays is equal to the index
     * of a cell, that cell is the "parent" of a cluster of cells. If
     * they are not, there is nothing for us to do. Easy!
     */
    if (f[tid] == tid) {
        /*
         * If we are a cluster owner, atomically claim a position in the
         * output array which we can write to.
         */
        const unsigned int id =
            vecmem::device_atomic_ref<unsigned int>(outi).fetch_add(1);

        /*
         * Now, we iterate over all other cells to check if they belong
         * to our cluster. Note that we can start at the current index
         * because no cell is ever a child of a cluster owned by a cell
         * with a higher ID.
         */
        float totalWeight = 0.;
        point2 mean{0., 0.}, var{0., 0.};
        const auto module_link = cells[tid + part.start].module_link;
        const cell_module& this_module = modules.at(module_link);
        for (unsigned short j = tid; j < part.size; j++) {
            /*
             * If the value of this cell is equal to our, that means it
             * is part of our cluster. In that case, we take its values
             * for position and add them to our accumulators.
             */
            if (f[j] == tid) {
                const unsigned int pos = j + part.start;
                const cell& this_cell = cells[pos].c;

                const float weight = traccc::detail::signal_cell_modelling(
                    this_cell.activation, this_module);

                if (weight > this_module.threshold) {
                    totalWeight += this_cell.activation;
                    const point2 cell_position =
                        traccc::detail::position_from_cell(this_cell,
                                                           this_module);
                    const point2 prev = mean;
                    const point2 diff = cell_position - prev;

                    mean = prev + (weight / totalWeight) * diff;
                    for (char i = 0; i < 2; ++i) {
                        var[i] = var[i] + weight * (diff[i]) *
                                              (cell_position[i] - mean[i]);
                    }
                }
            }
        }
        if (totalWeight > 0.) {
            for (char i = 0; i < 2; ++i) {
                var[i] /= totalWeight;
            }
            const auto pitch = this_module.pixel.get_pitch();
            var = var +
                  point2{pitch[0] * pitch[0] / 12, pitch[1] * pitch[1] / 12};
        }

        /*
         * Fill output vector with calculated cluster properties
         */
        out[id].meas.local = mean;
        out[id].meas.variance = var;
        out[id].module_link = module_link;
    }
}

}  // namespace traccc::device