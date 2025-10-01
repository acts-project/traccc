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

template <typename T>
TRACCC_HOST_DEVICE inline point2 position_from_cell(
    const edm::silicon_cell<T>& cell,
    const silicon_detector_description::const_device& det_descr,
	point2* cell_lower_position) {

    // The detector description for the module that the cell is on.
    const auto module_dd = det_descr.at(cell.module_index());
    // Calculate / construct the local cell position.
    scalar channel0_high = module_dd.reference_x() +
		(scalar{0.5f} + static_cast<scalar>(cell.channel0())) * module_dd.pitch_x();
    scalar channel1_high = module_dd.reference_y() +
		(scalar{0.5f} + static_cast<scalar>(cell.channel1())) * module_dd.pitch_y();

	if(cell_lower_position) {
		*cell_lower_position = 
		{channel0_high - module_dd.pitch_x(), channel1_high - module_dd.pitch_y()};
	}
	return {channel0_high, channel1_high};
}

template <typename T>
TRACCC_HOST_DEVICE inline void calc_cluster_properties(
    const edm::silicon_cluster<T>& cluster,
    const edm::silicon_cell_collection::const_device& cells,
    const silicon_detector_description::const_device& det_descr, point2& mean,
    point2& var, scalar& totalWeight) {

    point2 offset{0.f, 0.f};
    bool first_processed = false;

    // Loop over the cell indices of the cluster.
    for (const unsigned int cell_idx : cluster.cell_indices()) {

        // The cell object.
        const auto cell = cells.at(cell_idx);

        // Translate the cell readout value into a weight.
        const scalar weight =
            signal_cell_modelling(cell.activation(), det_descr);

        // Only consider cells over a minimum threshold.
        if (weight > det_descr.threshold().at(cell.module_index())) {

            // Update all output properties with this cell.
            totalWeight += weight;
            scalar weight_factor = weight / totalWeight;

            point2 cell_position = position_from_cell(cell, det_descr);

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

template <typename T>
TRACCC_HOST_DEVICE inline void fill_measurement(
    measurement_collection_types::device& measurements,
    measurement_collection_types::device::size_type index,
    const edm::silicon_cluster<T>& cluster,
    const edm::silicon_cell_collection::const_device& cells,
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
    assert(cluster.cell_indices().empty() == false);
    assert([&]() {
        const unsigned int module_idx =
            cells.module_index().at(cluster.cell_indices().front());
        for (const unsigned int cell_idx : cluster.cell_indices()) {
            if (cells.module_index().at(cell_idx) != module_idx) {
                return false;
            }
        }
        return true;
    }() == true);

    // Calculate the cluster properties
    scalar totalWeight = 0.f;
    point2 mean{0.f, 0.f}, var{0.f, 0.f};
    calc_cluster_properties(cluster, cells, det_descr, mean, var, totalWeight);

    assert(totalWeight > 0.f);

    // Access the measurement in question.
    measurement& m = measurements[index];

    // The index of the module the cluster is on.
    const unsigned int module_idx =
        cells.module_index().at(cluster.cell_indices().front());
    // The detector description for the module that the cluster is on.
    const auto module_dd = det_descr.at(module_idx);

    // Fill the measurement object.
    m.surface_link = module_dd.geometry_id();
    // normalize the cell position
    m.local = mean;

    // plus pitch^2 / 12
    const scalar pitch_x = module_dd.pitch_x();
    const scalar pitch_y = module_dd.pitch_y();
    m.variance = var + point2{pitch_x * pitch_x / static_cast<scalar>(12.),
                              pitch_y * pitch_y / static_cast<scalar>(12.)};

    // For the ambiguity resolution algorithm, give a unique measurement ID
    m.measurement_id = index;

    // Set the measurement dimensionality.
    m.meas_dim = module_dd.dimensions();

    // Set the measurement's subspace.
    m.subs = module_dd.subspace();
}

}  // namespace traccc::details
