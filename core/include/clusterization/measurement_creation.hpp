/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "definitions/primitives.hpp"
#include "edm/cell.hpp"
#include "edm/cluster.hpp"
#include "edm/measurement.hpp"
#include "utils/algorithm.hpp"

namespace traccc {

/// Connected component labeling.
struct measurement_creation
    : public algorithm<host_measurement_collection(const cluster_collection &,
                                                   const cell_module &)> {

    /// Callable operator for the connected component, based on one single
    /// module
    ///
    /// @param clusters are the input cells into the connected component, they
    /// are
    ///              per module and unordered
    ///
    /// C++20 piping interface
    ///
    /// @return a measurement collection - usually same size or sometime
    /// slightly smaller than the input
    host_measurement_collection operator()(
        const cluster_collection &c, const cell_module &m) const override {
        output_type measurements;
        this->operator()(c, m, measurements);
        return measurements;
    }

    /// Callable operator for the connected component, based on one single
    /// module
    ///
    /// @param clusters are the input cells into the connected component, they
    /// are
    ///              per module and unordered
    ///
    /// void interface
    ///
    /// @return a measurement collection - usually same size or sometime
    /// slightly smaller than the input
    void operator()(const cluster_collection &clusters,
                    const cell_module &module,
                    output_type &measurements) const {

        // Run the algorithm
        auto pitch = module.pixel.get_pitch();

        measurements.reserve(clusters.items.size());
        for (const auto &cluster : clusters.items) {
            scalar totalWeight = 0.;

            // To calculate the mean and variance with high numerical stability
            // we use a weighted variant of Welford's algorithm. This is a
            // single-pass online algorithm that works well for large numbers
            // of samples, as well as samples with very high values.
            //
            // To learn more about this algorithm please refer to:
            // [1] https://doi.org/10.1080/00401706.1962.10490022
            // [2] The Art of Computer Programming, Donald E. Knuth, second
            //     edition, chapter 4.2.2.
            point2 mean = {0., 0.}, var = {0., 0.};

            // Should not happen
            if (cluster.cells.empty()) {
                continue;
            }

            for (const auto &cell : cluster.cells) {
                scalar weight = clusters.signal(cell.activation);
                if (weight > clusters.threshold) {
                    totalWeight += cell.activation;
                    const point2 cell_position = clusters.position_from_cell(
                        cell.channel0, cell.channel1);

                    const point2 prev = mean;
                    const point2 diff = cell_position - prev;

                    for (std::size_t i = 0; i < 2; ++i) {
                        mean[i] = prev[i] + (weight / totalWeight) * (diff[i]);
                        var[i] = var[i] + weight * (diff[i]) *
                                              (cell_position[i] - mean[i]);
                    }
                }
            }
            if (totalWeight > 0.) {
                measurement m;
                // normalize the cell position
                m.local = mean;
                // normalize the variance
                m.variance[0] = var[0] / totalWeight;
                m.variance[1] = var[1] / totalWeight;
                // plus pitch^2 / 12
                m.variance = m.variance + point2{pitch[0] * pitch[0] / 12,
                                                 pitch[1] * pitch[1] / 12};
                // @todo add variance estimation
                measurements.push_back(std::move(m));
            }
        }
    }
};

}  // namespace traccc
