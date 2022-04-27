/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/clusterization/detail/sparse_ccl.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/cluster.hpp"
#include "traccc/utils/algorithm.hpp"

namespace traccc {

/// Connected component labelling
///
/// Note that the separation between the public and private interface is
/// only there in the class because the compilers can't automatically figure
/// out the "vector type" of the templated implementation, without adding a
/// lot of "internal knowledge" about the vector types into this piece of
/// code. So instead the public operators are specifically implemented for
/// the host- and device versions of the EDM, making use of a single
/// implementation internally.
///
class component_connection
    : public algorithm<host_cluster_container(const host_cell_collection&,
                                              const cell_module&)> {
    //   public algorithm<host_cluster_container(const device_cell_collection&,
    //                                           const cell_module&)> {
    public:
    /// Constructor for component_connection
    ///
    /// @param mr is the memory resource
    component_connection(vecmem::memory_resource& mr) : m_mr(mr) {}

    /// @name Operators to use in host code
    /// @{

    /// Callable operator for the connected component, based on one single
    /// module
    ///
    /// @param cells are the input cells into the connected component, they are
    ///              per module and unordered
    /// @param module The description of the module that the cells belong to
    ///
    /// c++20 piping interface:
    /// @return a cluster collection
    ///
    host_cluster_container operator()(
        const host_cell_collection& cells,
        const cell_module& module) const override {
        return this->operator()<vecmem::vector>(cells, module);
    }
    /// @}

    /// @name Operators to use in device code
    /// @{

    /// Callable operator for the connected component, based on one single
    /// module
    ///
    /// This version of the function is meant to be used in device code.
    ///
    /// @param cells are the input cells into the connected component, they are
    ///              per module and unordered
    /// @param module The description of the module that the cells belong to
    ///
    /// c++20 piping interface:
    /// @return a cluster collection
    ///
    // host_cluster_container operator()(
    //     const device_cell_collection& cells,
    //     const cell_module& module) const override {
    //     return this->operator()<vecmem::device_vector>(cells, module);
    // }

    private:
    /// Implementation for the public cell collection creation operators
    template <template <typename> class vector_type>
    host_cluster_container operator()(const cell_collection<vector_type>& cells,
                                      const cell_module& module) const {
        host_cluster_container clusters(&m_mr.get());
        this->operator()<vector_type>(cells, module, clusters);
        return clusters;
    }

    /// Implementation for the public cell collection creation operators
    template <template <typename> class vector_type>
    void operator()(const cell_collection<vector_type>& cells,
                    const cell_module& module,
                    host_cluster_container& clusters) const {

        // Run the algorithm
        unsigned int num_clusters = 0;
        vector_type<unsigned int> connected_cells(cells.size(), &m_mr.get());
        detail::sparse_ccl<vector_type, traccc::cell>(cells, connected_cells,
                                                      num_clusters);

        clusters.resize(num_clusters);
        for (auto& cl_id : clusters.get_headers()) {
            cl_id.module = module.module;
            cl_id.placement = module.placement;
        }

        auto& cluster_items = clusters.get_items();
        unsigned int icell = 0;
        for (auto cell_label : connected_cells) {
            auto cindex = static_cast<unsigned int>(cell_label - 1);
            if (cindex < cluster_items.size()) {
                cluster_items[cindex].push_back(cells[icell++]);
            }
        }
    }

    private:
    std::reference_wrapper<vecmem::memory_resource> m_mr;

};  // class component_connection

}  // namespace traccc
