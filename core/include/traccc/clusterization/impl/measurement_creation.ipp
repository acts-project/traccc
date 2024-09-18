/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <cassert>

namespace traccc::details {

TRACCC_HOST_DEVICE
inline scalar signal_cell_modelling(
    scalar signal_in, const silicon_detector_description::const_device&) {
    return signal_in;
}

TRACCC_HOST_DEVICE
inline vector2 position_from_cell(
    const unsigned int cell_idx,
    const edm::silicon_cell_collection::const_device& cells,
    const silicon_detector_description::const_device& det_descr) {

    // Retrieve the specific values based on module idx
    const unsigned int module_idx = cells.module_index().at(cell_idx);
    return {det_descr.reference_x().at(module_idx) +
                (scalar{0.5} + cells.channel0().at(cell_idx)) *
                    det_descr.pitch_x().at(module_idx),
            det_descr.reference_y().at(module_idx) +
                (scalar{0.5} + cells.channel1().at(cell_idx)) *
                    det_descr.pitch_y().at(module_idx)};
}

TRACCC_HOST_DEVICE inline void calc_cluster_properties(
    const unsigned int cluster_idx,
    const edm::silicon_cell_collection::const_device& cells,
    const edm::silicon_cluster_collection::const_device& clusters,
    const silicon_detector_description::const_device& det_descr, point2& mean,
    point2& var, scalar& totalWeight) {

    point2 offset{0., 0.};
    bool first_processed = false;

    // Loop over the cell indices of the cluster.
    for (const unsigned int cell_idx :
         clusters.cell_indices().at(cluster_idx)) {

        // Translate the cell readout value into a weight.
        const scalar weight =
            signal_cell_modelling(cells.activation().at(cell_idx), det_descr);

        // Only consider cells over a minimum threshold.
        if (weight >
            det_descr.threshold().at(cells.module_index().at(cell_idx))) {

            // Update all output properties with this cell.
            totalWeight += weight;
            scalar weight_factor = weight / totalWeight;

            point2 cell_position =
                position_from_cell(cell_idx, cells, det_descr);

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
    }

    mean = mean + offset;
}

TRACCC_HOST_DEVICE inline void fill_measurement(
    measurement_collection_types::device& measurements,
    const measurement_collection_types::device::size_type index,
    const edm::silicon_cell_collection::const_device& cells,
    const edm::silicon_cluster_collection::const_device& clusters,
    const silicon_detector_description::const_device& det_descr) {

    // To calculate the mean and variance with high numerical stability
    // we use a weighted variant of Welford's algorithm. This is a
    // single-pass online algorithm that works well for large numbers
    // of samples, as well as samples with very high values.
    //
    // To learn more about this algorithm please refer to:
    // [1] https://doi.org/10.1080/00401706.1962.10490022
    // [2] The Art of Computer Programming, Donald E. Knuth, second
    //     edition, chapter 4.2.2.

    // Security checks.
    assert(clusters.cell_indices()[index].empty() == false);
    assert([&]() {
        const unsigned int module_idx =
            cells.module_index().at(clusters.cell_indices()[index].front());
        for (const unsigned int cell_idx : clusters.cell_indices()[index]) {
            if (cells.module_index().at(cell_idx) != module_idx) {
                return false;
            }
        }
        return true;
    }() == true);

    // Calculate the cluster properties
    scalar totalWeight = 0.f;
    point2 mean{0.f, 0.f}, var{0.f, 0.f};
    calc_cluster_properties(index, cells, clusters, det_descr, mean, var,
                            totalWeight);

    assert(totalWeight > 0.f);

    // Access the measurement in question.
    measurement& m = measurements[index];

    // The index of the module the cluster is on.
    const unsigned int module_idx =
        cells.module_index().at(clusters.cell_indices().at(index).front());

    m.module_link = module_idx;
    m.surface_link = det_descr.geometry_id().at(module_idx);
    // normalize the cell position
    m.local = mean;

    // plus pitch^2 / 12
    const scalar pitch_x = det_descr.pitch_x().at(module_idx);
    const scalar pitch_y = det_descr.pitch_y().at(module_idx);
    m.variance = var + point2{pitch_x * pitch_x / static_cast<scalar>(12.),
                              pitch_y * pitch_y / static_cast<scalar>(12.)};

    // For the ambiguity resolution algorithm, give a unique measurement ID
    m.measurement_id = index;

    // Set the measurement dimensionality.
    m.meas_dim = det_descr.dimensions().at(module_idx);
}

}  // namespace traccc::details
