/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Detray Core include(s)
#include "detray/core/detail/multi_store.hpp"
#include "detray/core/detail/single_store.hpp"
#include "detray/core/detail/tuple_container.hpp"

// Vecmem include(s)
#include <vecmem/containers/device_vector.hpp>

namespace detray {

// Single store test
/// @{
using single_store_t = single_store<double, vecmem::vector>;
using single_store_dev_t = single_store<double, vecmem::device_vector>;
/// @}

// Tuple container test
/// @{
using tuple_cont_t = detail::tuple_container<dtuple, vecmem::vector<int>,
                                             vecmem::vector<double>>;
using tuple_cont_dev_t =
    detail::tuple_container<dtuple, vecmem::device_vector<int>,
                            vecmem::device_vector<double>>;
/// @}

// Regular multi store test (uses vectors as containers in every tuple element)
/// @{
using reg_multi_store_t =
    regular_multi_store<int, empty_context, dtuple, vecmem::vector, std::size_t,
                        float, double>;
using reg_multi_store_dev_t =
    regular_multi_store<int, empty_context, dtuple, vecmem::device_vector,
                        std::size_t, float, double>;
/// @}

/// Multi store test
/// @{

/// Test type that holds vecemem members and forces a hierarchical view/buffer
/// treatment
template <template <typename...> class vector_t = dvector>
struct test {

    using view_type = dmulti_view<dvector_view<int>, dvector_view<double>>;
    using const_view_type =
        dmulti_view<dvector_view<const int>, dvector_view<const double>>;
    using buffer_type =
        dmulti_buffer<dvector_buffer<int>, dvector_buffer<double>>;

    DETRAY_HOST explicit test(vecmem::memory_resource* mr)
        : first(mr), second(mr) {}

    template <concepts::device_view view_t>
    DETRAY_HOST_DEVICE explicit test(view_t v)
        : first(detail::get<0>(v.m_view)), second(detail::get<1>(v.m_view)) {}

    DETRAY_HOST view_type get_data() {
        return view_type{vecmem::get_data(first), vecmem::get_data(second)};
    }

    vector_t<int> first;
    vector_t<double> second;
};

using multi_store_t = multi_store<std::size_t, empty_context, dtuple,
                                  vecmem::vector<float>, test<vecmem::vector>>;
using multi_store_dev_t =
    multi_store<std::size_t, empty_context, dtuple,
                vecmem::device_vector<float>, test<vecmem::device_vector>>;
/// @}

void test_single_store(typename single_store_t::view_type store_view,
                       vecmem::data::vector_view<double> sum_data);

void test_tuple_container(typename tuple_cont_t::view_type store_view,
                          vecmem::data::vector_view<double> sum_data);

void test_reg_multi_store(typename reg_multi_store_t::view_type store_view,
                          vecmem::data::vector_view<double> sum_data);

void test_multi_store(typename multi_store_t::view_type store_view,
                      vecmem::data::vector_view<double> sum_data);

}  // namespace detray
