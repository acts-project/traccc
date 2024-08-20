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
    const vecmem::device_vector<details::index_t>& f, const unsigned int start,
    const unsigned int end, const unsigned short cid, measurement& out,
    vecmem::data::vector_view<unsigned int> cell_links,
    const unsigned int link) {
    vecmem::device_vector<unsigned int> cell_links_device(cell_links);

    /*
     * Now, we iterate over all other cells to check if they belong to our
     * cluster. Note that we can start at the current index because no cell is
     * ever a child of a cluster owned by a cell with a higher ID.
     *
     * Implemented here is a weighted version of Welford's algorithm. To read
     * more about this algorithm, see the following sources:
     *
     * [1] https://doi.org/10.1080/00401706.1962.10490022
     * [2] The Art of Computer Programming, Donald E. Knuth, second edition,
     * chapter 4.2.2.
     *
     * The core idea of Welford's algorithm is to use the recurrence relation
     *
     * $$\sigma^2_n = (1 - \frac{w_n}{W_n}) \sigma^2_{n-1} + \frac{w_n}{W_n}
     * (x_n - \mu_n) (x_n - \mu_{n-1})$$
     *
     * Which makes the algorithm less prone to catastrophic cancellation and
     * other unwanted effects. In addition, we offset the entire computation
     * by the first cell in the cluster, which brings the entire computation
     * closer to zero where floating point precision is higher. This relies on
     * the following:
     *
     * $$\mu(x_1, \ldots, x_n) = \mu(x_1 - C, \ldots, x_n - C) + C$$
     *
     * and
     *
     * $$\sigma^2(x_1, \ldots, x_n) = \sigma^2(x_1 - C, \ldots, x_n - C)$$
     */
    scalar totalWeight = 0.;
    point2 mean{0., 0.}, var{0., 0.}, offset{0., 0.};

    const auto module_link = cells[cid + start].module_link;
    const cell_module this_module = modules.at(module_link);
    const unsigned short partition_size = end - start;

    bool first_processed = false;

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
                scalar weight_factor = weight / totalWeight;

                point2 cell_position =
                    traccc::details::position_from_cell(this_cell, this_module);

                if (!first_processed) {
                    offset = cell_position;
                    first_processed = true;
                }

                cell_position = cell_position - offset;

                const point2 diff_old = cell_position - mean;
                mean = mean + diff_old * weight_factor;
                const point2 diff_new = cell_position - mean;

                var[0] = (1.f - weight_factor) * var[0] +
                         weight_factor * (diff_old[0] * diff_new[0]);
                var[1] = (1.f - weight_factor) * var[1] +
                         weight_factor * (diff_old[1] * diff_new[1]);
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

    const auto pitch = this_module.pixel.get_pitch();
    var = var + point2{pitch[0] * pitch[0] / static_cast<scalar>(12.),
                       pitch[1] * pitch[1] / static_cast<scalar>(12.)};

    /*
     * Fill output vector with calculated cluster properties
     */
    out.local = mean + offset;
    out.variance = var;
    out.surface_link = this_module.surface_link;
    out.module_link = module_link;
    // Set a unique identifier for the measurement.
    out.measurement_id = link;
    // Adjust the output object for 1D surfaces.
    if (this_module.pixel.dimension == 1) {
        out.meas_dim = 1;
        out.local[1] = 0.f;
    } else {
        out.meas_dim = 2;
    }
}

}  // namespace traccc::device
