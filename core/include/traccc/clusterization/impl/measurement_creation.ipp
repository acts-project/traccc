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
inline scalar signal_cell_modelling(scalar signal_in,
                                    const cell_module& /*mod*/) {
    return signal_in;
}

TRACCC_HOST_DEVICE
inline vector2 position_from_cell(const cell& cell, const cell_module& mod) {

    // Retrieve the specific values based on module idx
    return {mod.pixel.min_corner_x +
                (scalar{0.5} + cell.channel0) * mod.pixel.pitch_x,
            mod.pixel.min_corner_y +
                (scalar{0.5} + cell.channel1) * mod.pixel.pitch_y};
}

TRACCC_HOST_DEVICE inline void calc_cluster_properties(
    const cell_collection_types::const_device& cluster, const cell_module& mod,
    point2& mean, point2& var, scalar& totalWeight) {
    point2 offset{0., 0.};
    bool first_processed = false;

    // Loop over the cells of the cluster.
    for (const cell& cell : cluster) {

        // Translate the cell readout value into a weight.
        const scalar weight = signal_cell_modelling(cell.activation, mod);

        // Only consider cells over a minimum threshold.
        if (weight > mod.threshold) {
            totalWeight += weight;
            scalar weight_factor = weight / totalWeight;

            point2 cell_position = position_from_cell(cell, mod);

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
    std::size_t measurement_index,
    const cell_collection_types::const_device& cluster, const cell_module& mod,
    const unsigned int mod_link) {

    // To calculate the mean and variance with high numerical stability
    // we use a weighted variant of Welford's algorithm. This is a
    // single-pass online algorithm that works well for large numbers
    // of samples, as well as samples with very high values.
    //
    // To learn more about this algorithm please refer to:
    // [1] https://doi.org/10.1080/00401706.1962.10490022
    // [2] The Art of Computer Programming, Donald E. Knuth, second
    //     edition, chapter 4.2.2.

    // Calculate the cluster properties
    scalar totalWeight = 0.f;
    point2 mean{0.f, 0.f}, var{0.f, 0.f};
    calc_cluster_properties(cluster, mod, mean, var, totalWeight);

    assert(totalWeight > 0.f);

    // Access the measurement in question.
    measurement& m = measurements[measurement_index];

    m.module_link = mod_link;
    m.surface_link = mod.surface_link;
    // normalize the cell position
    m.local = mean;

    // plus pitch^2 / 12
    const auto pitch = mod.pixel.get_pitch();
    m.variance = var + point2{pitch[0] * pitch[0] / static_cast<scalar>(12.),
                              pitch[1] * pitch[1] / static_cast<scalar>(12.)};
    // @todo add variance estimation

    // For the ambiguity resolution algorithm, give a unique measurement ID
    m.measurement_id = measurement_index;

    // Adjust the measurement object for 1D surfaces.
    if (mod.pixel.dimension == 1) {
        m.meas_dim = 1;
        m.local[1] = 0.f;
    }
}

}  // namespace traccc::details
