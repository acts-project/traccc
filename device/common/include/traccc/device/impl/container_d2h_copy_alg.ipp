/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::device {

template <typename CONTAINER_TYPES>
container_d2h_copy_alg<CONTAINER_TYPES>::container_d2h_copy_alg(
    const memory_resource& mr, vecmem::copy& copy)
    : m_mr(mr), m_copy(copy) {}

template <typename CONTAINER_TYPES>
typename container_d2h_copy_alg<CONTAINER_TYPES>::output_type
container_d2h_copy_alg<CONTAINER_TYPES>::operator()(
    const typename CONTAINER_TYPES::const_view& input) const {

    // Create the result object.
    output_type result = make_output(input);

    // Perform the copy.
    m_copy(input.headers, result.get_headers(),
           vecmem::copy::type::device_to_host);
    m_copy(input.items, result.get_items(), vecmem::copy::type::device_to_host);

    // Return the host object.
    return result;
}

template <typename CONTAINER_TYPES>
typename container_d2h_copy_alg<CONTAINER_TYPES>::output_type
container_d2h_copy_alg<CONTAINER_TYPES>::operator()(
    const typename CONTAINER_TYPES::buffer& input) const {

    // Create the result object.
    output_type result = make_output(input);

    // Perform the copy.
    m_copy(input.headers, result.get_headers(),
           vecmem::copy::type::device_to_host);
    m_copy(input.items, result.get_items(), vecmem::copy::type::device_to_host);

    // Return the host object.
    return result;
}

template <typename CONTAINER_TYPES>
typename container_d2h_copy_alg<CONTAINER_TYPES>::output_type
container_d2h_copy_alg<CONTAINER_TYPES>::operator()(
    typename CONTAINER_TYPES::buffer&& input) const {

    // Call on the other function to do the work.
    const typename CONTAINER_TYPES::buffer& input_ref = input;
    return (*this)(input_ref);
}

template <typename CONTAINER_TYPES>
typename container_d2h_copy_alg<CONTAINER_TYPES>::output_type
container_d2h_copy_alg<CONTAINER_TYPES>::make_output(
    const typename CONTAINER_TYPES::const_view& input) const {

    // Decide what memory resource to use for the host container.
    vecmem::memory_resource* mr = m_mr.host ? m_mr.host : &(m_mr.main);

    // Create the result object, giving it the appropriate memory resource for
    // all of its elements.
    output_type result{input.items.m_size, mr};
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i].items =
            typename CONTAINER_TYPES::host::item_vector::value_type{mr};
    }

    // Return the output object.
    return result;
}

}  // namespace traccc::device
