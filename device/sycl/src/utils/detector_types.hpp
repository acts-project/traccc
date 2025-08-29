/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/definitions/primitives.hpp"
#include "traccc/geometry/detector.hpp"

namespace traccc::sycl {

/*
 * SYCL requires a little bit of extra massaging to make the kernel tags
 * work...
 */
struct default_detector_kernel_tag {};
struct telescope_detector_kernel_tag {};

template <typename T>
struct detector_tag_selector {};

template <>
struct detector_tag_selector<default_detector> {
    using type = default_detector_kernel_tag;
};

template <>
struct detector_tag_selector<telescope_detector> {
    using type = telescope_detector_kernel_tag;
};

template <typename T>
using detector_tag_selector_t = typename detector_tag_selector<T>::type;

template <typename T>
concept detector_tag_exists_for_backend =
    requires { typename detector_tag_selector_t<T>; };

template <typename>
struct detector_tag_existance_validator {};

template <typename... Ts>
struct detector_tag_existance_validator<std::tuple<Ts...>> {
    static constexpr bool value = (detector_tag_exists_for_backend<Ts> && ...);
};

static_assert(
    detector_tag_existance_validator<detector_type_list>::value,
    "Not all detector types registered for SYCL have an accompanying tag");

}  // namespace traccc::sycl
