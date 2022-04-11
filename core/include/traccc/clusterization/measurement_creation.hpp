/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/definitions/primitives.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/cluster.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/utils/algorithm.hpp"

namespace traccc {

/// Connected component labeling.
struct measurement_creation
    : public algorithm<host_measurement_collection(
          const host_cluster_container &, const cell_module &)> {
    public:
    /// Constructor for measurement_creation
    ///
    /// @param mr is the memory resource
    measurement_creation(vecmem::memory_resource &mr) : m_mr(mr) {}

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
        const host_cluster_container &c, const cell_module &m) const override {
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
    void operator()(const host_cluster_container &clusters,
                    const cell_module &module,
                    output_type &measurements) const {

        // Run the algorithm
        const auto &cl_id = clusters.at(0).header;
        auto pitch = get_pitch(cl_id.module_idx, cl_id.pixel);

        measurements.reserve(clusters.size());
        for (const auto &cluster : clusters.get_items()) {
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
            if (cluster.empty()) {
                continue;
            }

            for (const auto &cell : cluster) {
                scalar weight = signal_cell_modelling(cell.activation);
                if (weight > cl_id.threshold) {
                    totalWeight += cell.activation;
                    const point2 cell_position =
                        position_from_cell(cl_id.module_idx, cell, cl_id.pixel, cl_id.is_default);

                    const point2 prev = mean;
                    const point2 diff = cell_position - prev;

                    mean = prev + (weight / totalWeight) * diff;
                    for (std::size_t i = 0; i < 2; ++i) {
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

    private:
    std::reference_wrapper<vecmem::memory_resource> m_mr;
};

}  // namespace traccc
