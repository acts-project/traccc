/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::details {

TRACCC_HOST_DEVICE
inline scalar signal_cell_modelling(scalar signal_in,
                                    const cell_module& /*module*/) {
    return signal_in;
}

TRACCC_HOST_DEVICE
inline vector2 position_from_cell(const cell& cell, const cell_module& module) {

    // Retrieve the specific values based on module idx
    return {module.pixel.min_center_x + cell.channel0 * module.pixel.pitch_x,
            module.pixel.min_center_y + cell.channel1 * module.pixel.pitch_y};
}

TRACCC_HOST_DEVICE inline void calc_cluster_properties(
    const cell_collection_types::const_device& cluster,
    const cell_module& module, point2& mean, point2& var, scalar& totalWeight) {

    // Loop over the cells of the cluster.
    for (const cell& cell : cluster) {

        // Translate the cell readout value into a weight.
        const scalar weight = signal_cell_modelling(cell.activation, module);

        // Only consider cells over a minimum threshold.
        if (weight > module.threshold) {

            // Update all output properties with this cell.
            totalWeight += cell.activation;
            const point2 cell_position = position_from_cell(cell, module);
            const point2 prev = mean;
            const point2 diff = cell_position - prev;

            mean = prev + (weight / totalWeight) * diff;
            for (std::size_t i = 0; i < 2; ++i) {
                var[i] =
                    var[i] + weight * (diff[i]) * (cell_position[i] - mean[i]);
            }
        }
    }
}

TRACCC_HOST_DEVICE inline void fill_measurement(
    measurement_collection_types::device& measurements,
    std::size_t measurement_index,
    const cell_collection_types::const_device& cluster,
    const cell_module& module, const unsigned int module_link) {

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
    scalar totalWeight = 0.;
    point2 mean{0., 0.}, var{0., 0.};
    calc_cluster_properties(cluster, module, mean, var, totalWeight);

    if (totalWeight > 0.) {

        // Access the measurement in question.
        measurement& m = measurements[measurement_index];

        m.module_link = module_link;
        m.surface_link = module.surface_link;
        // normalize the cell position
        m.local = mean;
        // normalize the variance
        m.variance[0] = var[0] / totalWeight;
        m.variance[1] = var[1] / totalWeight;
        // plus pitch^2 / 12
        const auto pitch = module.pixel.get_pitch();
        m.variance =
            m.variance + point2{pitch[0] * pitch[0] / static_cast<scalar>(12.),
                                pitch[1] * pitch[1] / static_cast<scalar>(12.)};
        // @todo add variance estimation

        // For the ambiguity resolution algorithm, give a unique measurement ID
        m.measurement_id = measurement_index;

        // Adjust the measurement object for 1D surfaces.
        if (module.pixel.dimension == 1) {
            m.meas_dim = 1;
            m.local[1] = 0.f;
            m.variance[1] = 1000.f;
        }
    }
}

}  // namespace traccc::details
