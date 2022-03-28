/**
 * TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <type_traits>

// Project include(s).
#include <cuda_runtime.h>

#include <vecmem/memory/cuda/device_memory_resource.hpp>

#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/geometry/module_map.hpp"

namespace traccc::cuda {
template <typename, typename>
class module_map_view;

/**
 * @brief CUDA implementation of the module map, owning data.
 *
 * The existing module map is designed to be GPU-friendly, and this is the
 * realization of that idea. The CUDA module map comes in two flavours; the
 * first is the owning module map, which allocates memory on the device when
 * constructed and stores its data there. Then, we can create views from that.
 *
 * @tparam K Key type.
 * @tparam V Value type.
 */
template <typename K = geometry_id, typename V = transform3>
class module_map {
    public:
    /**
     * @brief The type of the internal tree nodes.
     */
    using node_t = typename ::traccc::module_map<K, V>::module_map_node;

    /**
     * @brief Construct a CUDA module map from an existing CPU module map.
     *
     * @note This is the only way to construct a CUDA module map.
     *
     * @param[in] input The existing module map to convert.
     */
    module_map(const ::traccc::module_map<K, V>& input,
               vecmem::memory_resource& mr)
        : m_n_nodes(input.m_nodes.size()),
          m_n_values(input.m_values.size()),
          m_nodes(vecmem::make_unique_alloc<node_t[]>(mr, m_n_nodes)),
          m_values(vecmem::make_unique_alloc<V[]>(mr, m_n_values)) {
        cudaMemcpy(m_nodes.get(), input.m_nodes.data(),
                   m_n_nodes * sizeof(node_t), cudaMemcpyHostToDevice);
        cudaMemcpy(m_values.get(), input.m_values.data(),
                   m_n_values * sizeof(V), cudaMemcpyHostToDevice);
    }

    private:
    /*
     * Store the number of nodes and values, since we cannot get this from the
     * vector anymore.
     */
    const std::size_t m_n_nodes;
    const std::size_t m_n_values;

    /**
     * @brief The internal storage of the nodes in our binary search tree.
     *
     * This follows the well-known formalism where the root node resides at
     * index 0, while for any node at position n, the left child is at index 2n
     * + 1, and the right child is at index 2n + 2.
     */
    vecmem::unique_alloc_ptr<node_t[]> m_nodes;

    /**
     * @brief This vector stores the values in a contiguous manner. Our nodes
     * keep indices in this array instead of pointers.
     */
    vecmem::unique_alloc_ptr<V[]> m_values;

    /*
     * Declare the view class as our friend, so that it can access our
     * pointers.
     */
    friend class module_map_view<K, V>;
};

/**
 * @brief CUDA implementation of the module map, non-owning view.
 *
 * This class represents a light-weight, non-owning, device-passable view of
 * a module map. It contains all the same data, but only has a view copy of the
 * pointers.
 *
 * @tparam K Key type.
 * @tparam V Value type.
 */
template <typename K = geometry_id, typename V = transform3>
class module_map_view {
    /**
     * @brief The type of the internal tree nodes.
     */
    public:
    using node_t = typename module_map<K, V>::node_t;

    /**
     * @brief Construct a new module map view object fom a module map.
     *
     * @param input The module map to create a view from.
     */
    module_map_view(const module_map<K, V>& input)
        : m_n_nodes(input.m_n_nodes),
          m_n_values(input.m_n_values),
          m_nodes(input.m_nodes.get()),
          m_values(input.m_values.get()) {}

    /**
     * @brief Find a given key in the map.
     *
     * @param[in] i The key to look-up.
     *
     * @return The value associated with the given key.
     *
     * @warning This method does no bounds checking, and will result in
     * undefined behaviour if the key does not exist in the map.
     */
    TRACCC_DEVICE const V* operator[](const K& i) const {
        unsigned int n = 0;

        while (true) {
            /*
             * For memory safety, if we are out of bounds we will exit.
             */
            if (n >= m_n_nodes) {
                return nullptr;
            }

            /*
             * Retrieve the current root node.
             */
            const node_t& node = m_nodes[n];

            /*
             * If the size is zero, it is essentially an invalid node (i.e. the
             * node does not exist).
             */
            if (node.size == 0) {
                return nullptr;
            }

            /*
             * If the value we are looking for is past the start of the current
             * node, there are three possibilities. Firstly, the value might be
             * in the current node. Secondly, the value might be in the right
             * child of the current node. Thirdly, the value might not be in the
             * map at all.
             */
            if (i >= node.start) {
                /*
                 * Next, we check if the value is within the range represented
                 * by the current node.
                 */
                if (i < node.start + node.size) {
                    /*
                     * Found it! Return a pointer to the value within the
                     * contiguous range.
                     */
                    return &m_values[node.index + (i - node.start)];
                } else {
                    /*
                     * Two possibilties remain, we need to check the right
                     * subtree.
                     */
                    n = 2 * n + 2;
                }
            }
            /*
             * If the value we want to find is less then the start of this node,
             * there are only two possibilities. Firstly, the value might be in
             * the left subtree, or the value might not be in the map at all.
             */
            else {
                n = 2 * n + 1;
            }
        }
    }

    /**
     * @brief Get the total number of modules in the module map.
     */
    TRACCC_DEVICE std::size_t size(void) const { return m_n_values; }

    /**
     * @brief Check if a module map contains a given module.
     */
    TRACCC_DEVICE bool contains(const K& i) const {
        return operator[](i) != nullptr;
    }

    /**
     * @brief Check whether a module map is empty.
     *
     * Probably not.
     */
    TRACCC_DEVICE bool empty(void) const { return m_n_values == 0; }

    const std::size_t m_n_nodes;
    const std::size_t m_n_values;
    const node_t* m_nodes;
    const V* m_values;
};
}  // namespace traccc::cuda
