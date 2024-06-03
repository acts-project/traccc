/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "traccc/clusterization/details/measurement_creation.hpp"

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void aggregate_cluster(
    const cell_collection_types::const_device& cells,
    const cell_module_collection_types::const_device& modules,
    const vecmem::data::vector_view<const unsigned short>& f_view,
    const unsigned int start, const unsigned int end, const unsigned short cid,
    measurement& out, vecmem::data::vector_view<unsigned int> cell_links,
    const unsigned int link) {

    const vecmem::device_vector<const unsigned short> f(f_view);
    vecmem::device_vector<unsigned int> cell_links_device(cell_links);

    /*
     * Now, we iterate over all other cells to check if they belong
     * to our cluster. Note that we can start at the current index
     * because no cell is ever a child of a cluster owned by a cell
     * with a higher ID.
     */
    scalar totalWeight = 0.;
    point2 mean{0., 0.}, var{0., 0.};
    const auto module_link = cells[cid + start].module_link;
    const cell_module this_module = modules.at(module_link);
    const unsigned short partition_size = end - start;

    channel_id maxChannel1 = std::numeric_limits<channel_id>::min();

    for (unsigned short j = cid; j < partition_size; j++) {

        const unsigned int pos = j + start;
        /*
         * Terminate the process earlier if we have reached a cell sufficiently
         * in a different module.
         */
        if (cells[pos].module_link != module_link) {
            break;
        }

        const cell this_cell = cells[pos];

        /*
         * If the value of this cell is equal to our, that means it
         * is part of our cluster. In that case, we take its values
         * for position and add them to our accumulators.
         */
        if (f.at(j) == cid) {

            if (this_cell.channel1 > maxChannel1) {
                maxChannel1 = this_cell.channel1;
            }

            const scalar weight = traccc::details::signal_cell_modelling(
                this_cell.activation, this_module);

            if (weight > this_module.threshold) {
                totalWeight += weight;
                const point2 cell_position =
                    traccc::details::position_from_cell(this_cell, this_module);
                const point2 prev = mean;
                const point2 diff = cell_position - prev;

                mean = prev + (weight / totalWeight) * diff;
                for (char i = 0; i < 2; ++i) {
                    var[i] += weight * (diff[i]) * (cell_position[i] - mean[i]);
                }
            }

            cell_links_device.at(pos) = link;
        }

        /*
         * Terminate the process earlier if we have reached a cell sufficiently
         * far away from the cluster in the dominant axis.
         */
        if (this_cell.channel1 > maxChannel1 + 1) {
            break;
        }
    }
    if (totalWeight > static_cast<scalar>(0.)) {
#pragma unroll
        for (char i = 0; i < 2; ++i) {
            var[i] /= totalWeight;
        }
        const auto pitch = this_module.pixel.get_pitch();
        var = var + point2{pitch[0] * pitch[0] / static_cast<scalar>(12.),
                           pitch[1] * pitch[1] / static_cast<scalar>(12.)};
    }

    /*
     * Fill output vector with calculated cluster properties
     */
    out.local = mean;
    out.variance = var;
    out.surface_link = this_module.surface_link;
    out.module_link = module_link;
    // Set a unique identifier for the measurement.
    out.measurement_id = link;
    // Adjust the output object for 1D surfaces.
    if (this_module.pixel.dimension == 1) {
        out.meas_dim = 1;
        out.local[1] = 0.f;
        out.variance[1] = this_module.pixel.variance_y;
    } else {
        out.meas_dim = 2;
    }
}

}  // namespace traccc::device
