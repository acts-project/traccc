/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// System include(s).
#include <cassert>

namespace vecmem {

template <typename TYPE>
data::jagged_vector_data<TYPE> get_data(jagged_vector<TYPE>& vec,
                                        memory_resource* resource) {

    // Get the size of the "outer vector".
    const std::size_t size = vec.size();

    // Construct the object to be returned.
    data::jagged_vector_data<TYPE> result(
        size,
        (resource != nullptr ? *resource : *(vec.get_allocator().resource())));

    // Helper local type definition(s).
    typedef typename data::jagged_vector_data<TYPE>::value_type value_type;
    typedef typename value_type::size_type size_type;

    // Fill the result object with information.
    for (std::size_t i = 0; i < size; ++i) {
        result.host_ptr()[i] =
            value_type(static_cast<size_type>(vec[i].size()), vec[i].data());
    }

    // Return the created object.
    return result;
}

template <typename TYPE, typename ALLOC1, typename ALLOC2>
data::jagged_vector_data<TYPE> get_data(
    std::vector<std::vector<TYPE, ALLOC1>, ALLOC2>& vec,
    memory_resource* resource) {

    // This function needs a non-null memory resource pointer.
    assert(resource != nullptr);

    // Get the size of the "outer vector".
    const std::size_t size = vec.size();

    // Construct the object to be returned.
    data::jagged_vector_data<TYPE> result(size, *resource);

    // Helper local type definition(s).
    typedef typename data::jagged_vector_data<TYPE>::value_type value_type;
    typedef typename value_type::size_type size_type;

    // Fill the result object with information.
    for (std::size_t i = 0; i < size; ++i) {
        result.host_ptr()[i] =
            value_type(static_cast<size_type>(vec[i].size()), vec[i].data());
    }

    // Return the created object.
    return result;
}

template <typename TYPE>
data::jagged_vector_data<const TYPE> get_data(const jagged_vector<TYPE>& vec,
                                              memory_resource* resource) {

    // Get the size of the "outer vector".
    const std::size_t size = vec.size();

    // Construct the object to be returned.
    data::jagged_vector_data<const TYPE> result(
        size,
        (resource != nullptr ? *resource : *(vec.get_allocator().resource())));

    // Helper local type definition(s).
    typedef
        typename data::jagged_vector_data<const TYPE>::value_type value_type;
    typedef typename value_type::size_type size_type;

    // Fill the result object with information.
    for (std::size_t i = 0; i < size; ++i) {
        result.host_ptr()[i] =
            value_type(static_cast<size_type>(vec[i].size()), vec[i].data());
    }

    // Return the created object.
    return result;
}

template <typename TYPE, typename ALLOC1, typename ALLOC2>
data::jagged_vector_data<const TYPE> get_data(
    const std::vector<std::vector<TYPE, ALLOC1>, ALLOC2>& vec,
    memory_resource* resource) {

    // This function needs a non-null memory resource pointer.
    assert(resource != nullptr);

    // Get the size of the "outer vector".
    const std::size_t size = vec.size();

    // Construct the object to be returned.
    data::jagged_vector_data<const TYPE> result(size, *resource);

    // Helper local type definition(s).
    typedef
        typename data::jagged_vector_data<const TYPE>::value_type value_type;
    typedef typename value_type::size_type size_type;

    // Fill the result object with information.
    for (std::size_t i = 0; i < size; ++i) {
        result.host_ptr()[i] =
            value_type(static_cast<size_type>(vec[i].size()), vec[i].data());
    }

    // Return the created object.
    return result;
}

}  // namespace vecmem
