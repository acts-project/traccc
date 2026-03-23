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

TRACCC_HOST_DEVICE inline void aggregate_cluster(
    const clustering_config& cfg,
    const edm::silicon_cell_collection::const_device& cells,
    const detector_design_description::const_device& det_desc,
    const detector_conditions_description::const_device& det_cond,
    const vecmem::device_vector<details::index_t>& f, const unsigned int start,
    const unsigned int end, const unsigned short cid,
    edm::measurement_collection<default_algebra>::device::proxy_type out,
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
    point2 mean{0.f, 0.f}, var{0.f, 0.f}, offset{0.f, 0.f}, width{0.f, 0.f},
        pitch{0.f, 0.f};

    unsigned int min_channel0 = std::numeric_limits<unsigned int>::max();
    unsigned int max_channel0 = std::numeric_limits<unsigned int>::lowest();
    unsigned int min_channel1 = std::numeric_limits<unsigned int>::max();
    unsigned int max_channel1 = std::numeric_limits<unsigned int>::lowest();

    const unsigned int module_idx = cells.module_index().at(cid + start);
    const auto module_cd = det_cond.at(module_idx);
    const unsigned int design_idx = module_cd.module_to_design_id();
    const auto module_dd = det_desc.at(design_idx);

    unsigned int tmp_cluster_size = 0;

    const auto partition_size = static_cast<unsigned short>(end - start);

    bool first_processed = false;

    channel_id maxChannel1 = std::numeric_limits<channel_id>::min();

    for (unsigned short j = cid; j < partition_size; j++) {

        const unsigned int pos = j + start;
        const edm::silicon_cell cell = cells.at(pos);

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

            const scalar weight = cell.activation();

            totalWeight += weight;
            scalar weight_factor = weight / totalWeight;

            point2 cell_position =
                traccc::details::position_from_cell(cell, module_dd);

            min_channel0 = std::min(min_channel0, cell.channel0());
            min_channel1 = std::min(min_channel1, cell.channel1());
            max_channel0 = std::max(max_channel0, cell.channel0());
            max_channel1 = std::max(max_channel1, cell.channel1());

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

            cell_links_device.at(pos) = link;
            tmp_cluster_size++;

            if (disjoint_set.capacity()) {
                disjoint_set.at(pos) = link;
            }

            if (cluster_size.has_value()) {
                (*cluster_size).get() = tmp_cluster_size;
            }
        }

        /*
         * Terminate the process earlier if we have reached a cell sufficiently
         * far away from the cluster in the dominant axis.
         */
        if (cell.channel1() > maxChannel1 + 1) {
            break;
        }
    }

    unsigned int delta0 = (max_channel0 - min_channel0) + 1;
    unsigned int delta1 = (max_channel1 - min_channel1) + 1;

    vector2 cluster_lower_position = {
        (module_dd.bin_edges_x()).at(min_channel0),
        (module_dd.bin_edges_y()).at(min_channel1)};

    vector2 cluster_upper_position = {
        (module_dd.bin_edges_x()).at(max_channel0 + 1),
        (module_dd.bin_edges_y()).at(max_channel1 + 1)};

    width[0] = cluster_upper_position[0] - cluster_lower_position[0];
    width[1] = cluster_upper_position[1] - cluster_lower_position[1];

    pitch[0] = width[0] / static_cast<float>(delta0);
    pitch[1] = width[1] / static_cast<float>(delta1);

    var = var + point2{pitch[0] * pitch[0] / static_cast<scalar>(12.),
                       pitch[1] * pitch[1] / static_cast<scalar>(12.)};

    point2 diameter = {width[0] / static_cast<scalar>(delta0),
                       width[1] / static_cast<scalar>(delta1)};

    /*
     * Fill output vector with calculated cluster properties
     */
    out.local_position() = mean + offset + module_cd.measurement_translation();
    out.local_variance() = var;
    out.surface_link() = module_cd.geometry_id();
    // Set a unique identifier for the measurement.
    out.identifier() = link;
    // Set the dimensionality of the measurement.
    out.dimensions() = module_dd.dimensions();
    // Set the measurement's subspace.
    out.subspace() = module_dd.subspace();
    // Set the index of the cluster that would be created for this measurement
    out.cluster_index() = link;

    if (cfg.diameter_strategy == clustering_diameter_strategy::CHANNEL0) {
        out.diameter() = diameter[0];
    } else if (cfg.diameter_strategy ==
               clustering_diameter_strategy::CHANNEL1) {
        out.diameter() = diameter[1];
    } else if (cfg.diameter_strategy == clustering_diameter_strategy::MAXIMUM) {
        out.diameter() = std::max(diameter[0], diameter[1]);
    } else if (cfg.diameter_strategy ==
               clustering_diameter_strategy::DIAGONAL) {
        out.diameter() =
            math::sqrt(diameter[0] * diameter[0] + diameter[1] * diameter[1]);
    }
}

}  // namespace traccc::device
