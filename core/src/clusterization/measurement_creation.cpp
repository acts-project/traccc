/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/clusterization/measurement_creation.hpp"

#include "traccc/clusterization/detail/measurement_creation_helper.hpp"
#include "traccc/definitions/primitives.hpp"

namespace traccc {

measurement_creation::measurement_creation(vecmem::memory_resource &mr)
    : m_mr(mr) {}

measurement_creation::output_type measurement_creation::operator()(
    const cluster_container_types::host &clusters,
    const cell_module &module) const {

    // Create the result object.
    output_type result(&(m_mr.get()));

    // If there are no clusters for this detector module, exit early.
    if (clusters.size() == 0) {
        return result;
    }

    // Reserve memory for one measurement per cluster.
    result.reserve(clusters.size());

    // Get values used for every cluster's processing.
    const vector2 pitch = module.pixel.get_pitch();
    const cluster_id &cl_id = clusters[0].header;

    // Process the clusters one-by-one.
    for (std::size_t i = 0; i < clusters.size(); ++i) {

        // To calculate the mean and variance with high numerical stability
        // we use a weighted variant of Welford's algorithm. This is a
        // single-pass online algorithm that works well for large numbers
        // of samples, as well as samples with very high values.
        //
        // To learn more about this algorithm please refer to:
        // [1] https://doi.org/10.1080/00401706.1962.10490022
        // [2] The Art of Computer Programming, Donald E. Knuth, second
        //     edition, chapter 4.2.2.

        // Get the cluster.
        const cluster_id &cl_id = clusters.at(i).header;
        cluster_container_types::host::item_vector::const_reference cluster =
            clusters.at(i).items;
        cluster_container_types::const_device::item_vector::value_type
            cluster_device(vecmem::get_data(cluster));

        // A security check.
        assert(cluster.empty() == false);

        // Calculate the cluster properties
        point2 mean{0., 0.}, var{0., 0.};
        scalar totalWeight = 0.;
        detail::calc_cluster_properties(cluster_device, cl_id, mean, var,
                                        totalWeight);

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
            result.push_back(std::move(m));
        }
    }

    // Return the measurements.
    return result;
}

}  // namespace traccc
