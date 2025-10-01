/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "traccc/clusterization/details/measurement_creation.hpp"

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void aggregate_cluster(
    const clustering_config& cfg,
    const edm::silicon_cell_collection::const_device& cells,
    const silicon_detector_description::const_device& det_descr,
    const vecmem::device_vector<details::index_t>& f, const unsigned int start,
    const unsigned int end, const unsigned short cid, measurement& out,
    vecmem::data::vector_view<unsigned int> cell_links, const unsigned int link,
    vecmem::device_vector<unsigned int>& disjoint_set,
    std::optional<std::reference_wrapper<unsigned int>> cluster_size) {
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
    scalar totalWeight = 0.f;
    point2 mean{0.f, 0.f}, var{0.f, 0.f}, offset{0.f, 0.f};

    //std::...::min here gives 0 
    scalar min_channel0 = std::numeric_limits<scalar>::max();
    scalar max_channel0 = -1*std::numeric_limits<scalar>::max();
    scalar min_channel1 = std::numeric_limits<scalar>::max();
    scalar max_channel1 = -1*std::numeric_limits<scalar>::max();
    
    const unsigned int module_idx = cells.module_index().at(cid + start);
    const auto module_descr = det_descr.at(module_idx);
    const auto partition_size = static_cast<unsigned short>(end - start);
    unsigned int tmp_cluster_size = 0;

    bool first_processed = false;

    channel_id maxChannel1 = std::numeric_limits<channel_id>::min();

    for (unsigned short j = cid; j < partition_size; j++) {

        const unsigned int pos = j + start;
        const auto cell = cells.at(pos);

        /*
         * Terminate the process earlier if we have reached a cell sufficiently
         * in a different module.
         */
        if (cell.module_index() != module_idx) {
            break;
        }

        /*
         * If the value of this cell is equal to our, that means it
         * is part of our cluster. In that case, we take its values
         * for position and add them to our accumulators.
         */
        if (f.at(j) == cid) {

            if (cell.channel1() > maxChannel1) {
                maxChannel1 = cell.channel1();
            }

            const scalar weight = traccc::details::signal_cell_modelling(
                cell.activation(), det_descr);

            if (weight > module_descr.threshold()) {
                totalWeight += weight;
                scalar weight_factor = weight / totalWeight;
 
                point2 cell_lower_position = {0,0};
                point2 cell_position = 
                    traccc::details::position_from_cell(cell, det_descr, &cell_lower_position);

                //calculated from the most-extreme cell edges
                min_channel0 = std::min(min_channel0, cell_lower_position[0]);
                max_channel0 = std::max(max_channel0, cell_position[0]);
                min_channel1 = std::min(min_channel1, cell_lower_position[1]);
                max_channel1 = std::max(max_channel1, cell_position[1]);

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

            tmp_cluster_size++;

            if (disjoint_set.capacity()) {
                disjoint_set.at(pos) = link;
            }
        }

        /*
         * Terminate the process earlier if we have reached a cell sufficiently
         * far away from the cluster in the dominant axis.
         */
        if (cell.channel1() > maxChannel1 + 1) {
            break;
        }

        if (cluster_size.has_value()) {
            (*cluster_size).get() = tmp_cluster_size;
        }
    }

    var = var + point2{module_descr.pitch_x() * module_descr.pitch_x() /
                           static_cast<scalar>(12.),
                       module_descr.pitch_y() * module_descr.pitch_y() /
                           static_cast<scalar>(12.)};

    /*
     * Fill output vector with calculated cluster properties
     */
    out.local = mean + offset + module_descr.measurement_translation();
    out.variance = var;
    out.surface_link = module_descr.geometry_id();
    // Set a unique identifier for the measurement.
    out.measurement_id = link;
    // Set the dimensionality of the measurement.
    out.meas_dim = module_descr.dimensions();
    // Set the measurement's subspace.
    out.subs = module_descr.subspace();

    scalar delta0 = max_channel0 - min_channel0;
    scalar delta1 = max_channel1 - min_channel1;
    if (cfg.diameter_strategy == clustering_diameter_strategy::CHANNEL0) {
        out.diameter = delta0;
    } else if (cfg.diameter_strategy ==
               clustering_diameter_strategy::CHANNEL1) {
        out.diameter = delta1;
    } else if (cfg.diameter_strategy == clustering_diameter_strategy::MAXIMUM) {
        out.diameter = std::max(delta0, delta1);
    } else if (cfg.diameter_strategy ==
               clustering_diameter_strategy::DIAGONAL) {
        out.diameter = math::sqrt(delta0 * delta0 + delta1 * delta1);
    }
}

}  // namespace traccc::device
