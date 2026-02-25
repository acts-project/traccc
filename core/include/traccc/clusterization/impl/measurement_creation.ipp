/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <cassert>

namespace traccc::details {

template <typename T>
TRACCC_HOST_DEVICE inline scalar signal_cell_modelling(
    scalar signal_in, const traccc::detector_conditions_description_interface<T>&) {
    return signal_in;
}

template <typename TCell, typename TDesign>
TRACCC_HOST_DEVICE inline vector2 position_from_cell(
    const edm::silicon_cell<TCell>& cell,
    const traccc::detector_design_description_interface<TDesign>& module_dd,
    vector2* cell_width) {

    // The detector description for the module that the cell is on.
    // const detector_conditions_description_interface module_cd =
    //     det_cond.at(cell.module_index());
    // const unsigned int design_idx =
    //     module_cd.module_to_design_id();
    // const detector_design_description_interface module_dd =
    //     det_descr.at(design_idx);

    // Calculate / construct the local cell position.
    vector2 cell_lower_position = {
        (module_dd.bin_edges_x()).at(cell.channel0()),
        (module_dd.bin_edges_y()).at(cell.channel1()) };

    vector2 cell_upper_position = {
        (module_dd.bin_edges_x()).at(cell.channel0()+1),
        (module_dd.bin_edges_y()).at(cell.channel1()+1) };

    vector2 cell_middle_position = {
        0.5*(cell_upper_position[0] + cell_lower_position[0]),
        0.5*(cell_upper_position[1] + cell_lower_position[1]),
    };

    *cell_width = {
        cell_upper_position[0] - cell_lower_position[0],
        cell_upper_position[1] - cell_lower_position[1],
    };

    return cell_middle_position;
}

template <typename T, typename TDesign, typename TCond>
TRACCC_HOST_DEVICE inline void calc_cluster_properties(
    const edm::silicon_cluster<T>& cluster,
    const edm::silicon_cell_collection::const_device& cells,
    const traccc::detector_design_description_interface<TDesign>& module_dd,
    const traccc::detector_conditions_description_interface<TCond>& module_cd,
    point2& mean, point2& var, scalar& totalWeight) {

    point2 offset{0.f, 0.f};
    bool first_processed = false;
    std::vector<int> phiIndices;
    std::vector<int> etaIndices;

    // Loop over the cell indices of the cluster.
    for (const unsigned int cell_idx : cluster.cell_indices()) {

        // The cell object.
        const edm::silicon_cell cell = cells.at(cell_idx);

        phiIndices.push_back(cell.channel0());
        etaIndices.push_back(cell.channel1());

        // Translate the cell readout value into a weight.
        const scalar weight =
            signal_cell_modelling(cell.activation(), module_cd);

        // Only consider cells over a minimum threshold.
        if (weight > module_cd.threshold()) {

            // Update all output properties with this cell.
            totalWeight += weight;
            scalar weight_factor = weight / totalWeight;

            point2 cell_width = {0.f,0.f};
            point2 cell_position = position_from_cell(cell, module_dd, &cell_width);

            if (!first_processed) {
                offset = cell_position;
                first_processed = true;
            }

            cell_position = cell_position - offset;

            const point2 diff_old = cell_position - mean;
            mean = mean + diff_old * weight_factor;
            const point2 diff_new = cell_position - mean;

            var[0] = var[0] + cell_width[0];
            var[1] = var[1] + cell_width[1];

        }
    }

    var[0] = var[0] / (std::ranges::max(phiIndices) - std::ranges::min(phiIndices) +1);
    var[1] = var[1] / (std::ranges::max(etaIndices) - std::ranges::min(etaIndices) +1);

    mean = mean + offset;
}

template <typename T1, typename T2>
TRACCC_HOST_DEVICE inline void fill_measurement(
    edm::measurement<T1>& measurement, const edm::silicon_cluster<T2>& cluster,
    const unsigned int index,
    const edm::silicon_cell_collection::const_device& cells,
    const detector_design_description::const_device& det_descr,
    const detector_conditions_description::const_device& det_cond) {

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


    // The index of the module the cluster is on.
    const unsigned int module_idx =
        cells.module_index().at(cluster.cell_indices().front());

    // The detector description for the module that the cluster is on.
    const detector_conditions_description_interface module_cd =
        det_cond.at(module_idx);
    const unsigned int design_idx =
        module_cd.module_to_design_id();
    const detector_design_description_interface module_dd =
        det_descr.at(design_idx);

    // Calculate the cluster properties
    scalar totalWeight = 0.f;
    point2 mean{0.f, 0.f}, var{0.f, 0.f};
    calc_cluster_properties(cluster, cells, module_dd, module_cd, mean, var, totalWeight);

    assert(totalWeight > 0.f);

    // Fill the measurement object.
    measurement.surface_link() = module_cd.geometry_id();

    // apply lorentz shift to the cell position
    std::array<float, 2> shift = module_cd.measurement_translation();
    measurement.local_position() = {mean[0]+shift[0],mean[1]+shift[1]};

    // plus pitch^2 / 12
    // const scalar pitch_x = 0.05; //module_dd.pitch_x();
    // const scalar pitch_y = 0.05; //module_dd.pitch_y();
    measurement.local_variance() =
              point2{var[0] * var[0] / static_cast<scalar>(12.),
                     var[1] * var[1] / static_cast<scalar>(12.)};

    // For the ambiguity resolution algorithm, give a unique measurement ID
    measurement.identifier() = index;
    measurement.cluster_index() = index;

    // Set the measurement dimensionality.
    measurement.dimensions() = module_dd.dimensions();

    // Set the measurement's subspace.
    measurement.subspace() = module_dd.subspace();

    // Save the index of the cluster that produced this measurement
    measurement.cluster_index() = static_cast<unsigned int>(index);
}

}  // namespace traccc::details
