/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/containers/data/jagged_vector_view.hpp"
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/memory/unique_ptr.hpp"

// System include(s).
#include <memory>

namespace vecmem {
namespace data {

/**
 * @brief A data wrapper for jagged vectors.
 *
 * This class constructs the relevant administrative data from a vector of
 * vectors, and is designed to be later turned into a @c jagged_vector_view
 * object.
 */
template <typename T>
class jagged_vector_data : public jagged_vector_view<T> {

public:
    /// Type of the base class
    using base_type = jagged_vector_view<T>;
    /// Use the base class's @c size_type
    typedef typename base_type::size_type size_type;
    /// Use the base class's @c value_type
    typedef typename base_type::value_type value_type;

    /// Default constructor
    jagged_vector_data();
    /**
     * @brief Construct jagged vector data from raw information
     *
     * This class converts from std vectors (or rather, vecmem::vectors) to
     * a jagged vector data.
     *
     * @param[in] size Size of the "outer vector"
     * @param[in] mem The memory resource to manage the internal state
     */
    jagged_vector_data(size_type size, memory_resource& mem);
    /// Move constructor
    jagged_vector_data(jagged_vector_data&&) = default;

    /// Move assignment
    jagged_vector_data& operator=(jagged_vector_data&&) = default;

private:
    /// Data object owning the allocated memory
    vecmem::unique_alloc_ptr<value_type[]> m_memory;

};  // class jagged_vector_data

}  // namespace data

/// Helper function creating a @c vecmem::data::jagged_vector_view object
template <typename TYPE>
data::jagged_vector_view<TYPE>& get_data(data::jagged_vector_data<TYPE>& data);

/// Helper function creating a @c vecmem::data::jagged_vector_view object
template <typename TYPE>
const data::jagged_vector_view<TYPE>& get_data(
    const data::jagged_vector_data<TYPE>& data);

}  // namespace vecmem

// Include the implementation.
#include "vecmem/containers/impl/jagged_vector_data.ipp"
