/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <utility>

#include <covfie/core/utility/backend_traits.hpp>

namespace covfie {
template <typename... Ts>
class parameter_pack
{
};

template <>
class parameter_pack<>
{
public:
    parameter_pack()
    {
    }
};

template <typename T, typename... Ts>
class parameter_pack<T, Ts...>
{
public:
    parameter_pack(T && _x, Ts &&... _xs)
        : x(std::forward<T>(_x))
        , xs(std::forward<Ts>(_xs)...)
    {
    }

    T x;
    parameter_pack<Ts...> xs;
};

template <typename... Ts>
parameter_pack<Ts...> make_parameter_pack(Ts &&... args)
{
    return parameter_pack<Ts...>(std::forward<Ts>(args)...);
}

// WARNING: These functions are automatically generated. Best not edit them
// by hand.

template <
    typename F,
    std::enable_if_t<
        utility::backend_depth<typename F::backend_t>::value == 1,
        bool> = true>
auto make_parameter_pack_for(typename utility::nth_backend<
                             typename F::backend_t,
                             0>::type::configuration_t && a0)
{
    return make_parameter_pack(
        std::forward<typename utility::nth_backend<typename F::backend_t, 0>::
                         type::configuration_t>(a0)
    );
}

template <
    typename F,
    std::enable_if_t<
        utility::backend_depth<typename F::backend_t>::value == 2,
        bool> = true>
auto make_parameter_pack_for(
    typename utility::nth_backend<typename F::backend_t, 0>::type::
        configuration_t && a0,
    typename utility::nth_backend<typename F::backend_t, 1>::type::
        configuration_t && a1
)
{
    return make_parameter_pack(
        std::forward<typename utility::nth_backend<typename F::backend_t, 0>::
                         type::configuration_t>(a0),
        std::forward<typename utility::nth_backend<typename F::backend_t, 1>::
                         type::configuration_t>(a1)
    );
}

template <
    typename F,
    std::enable_if_t<
        utility::backend_depth<typename F::backend_t>::value == 3,
        bool> = true>
auto make_parameter_pack_for(
    typename utility::nth_backend<typename F::backend_t, 0>::type::
        configuration_t && a0,
    typename utility::nth_backend<typename F::backend_t, 1>::type::
        configuration_t && a1,
    typename utility::nth_backend<typename F::backend_t, 2>::type::
        configuration_t && a2
)
{
    return make_parameter_pack(
        std::forward<typename utility::nth_backend<typename F::backend_t, 0>::
                         type::configuration_t>(a0),
        std::forward<typename utility::nth_backend<typename F::backend_t, 1>::
                         type::configuration_t>(a1),
        std::forward<typename utility::nth_backend<typename F::backend_t, 2>::
                         type::configuration_t>(a2)
    );
}

template <
    typename F,
    std::enable_if_t<
        utility::backend_depth<typename F::backend_t>::value == 4,
        bool> = true>
auto make_parameter_pack_for(
    typename utility::nth_backend<typename F::backend_t, 0>::type::
        configuration_t && a0,
    typename utility::nth_backend<typename F::backend_t, 1>::type::
        configuration_t && a1,
    typename utility::nth_backend<typename F::backend_t, 2>::type::
        configuration_t && a2,
    typename utility::nth_backend<typename F::backend_t, 3>::type::
        configuration_t && a3
)
{
    return make_parameter_pack(
        std::forward<typename utility::nth_backend<typename F::backend_t, 0>::
                         type::configuration_t>(a0),
        std::forward<typename utility::nth_backend<typename F::backend_t, 1>::
                         type::configuration_t>(a1),
        std::forward<typename utility::nth_backend<typename F::backend_t, 2>::
                         type::configuration_t>(a2),
        std::forward<typename utility::nth_backend<typename F::backend_t, 3>::
                         type::configuration_t>(a3)
    );
}

template <
    typename F,
    std::enable_if_t<
        utility::backend_depth<typename F::backend_t>::value == 5,
        bool> = true>
auto make_parameter_pack_for(
    typename utility::nth_backend<typename F::backend_t, 0>::type::
        configuration_t && a0,
    typename utility::nth_backend<typename F::backend_t, 1>::type::
        configuration_t && a1,
    typename utility::nth_backend<typename F::backend_t, 2>::type::
        configuration_t && a2,
    typename utility::nth_backend<typename F::backend_t, 3>::type::
        configuration_t && a3,
    typename utility::nth_backend<typename F::backend_t, 4>::type::
        configuration_t && a4
)
{
    return make_parameter_pack(
        std::forward<typename utility::nth_backend<typename F::backend_t, 0>::
                         type::configuration_t>(a0),
        std::forward<typename utility::nth_backend<typename F::backend_t, 1>::
                         type::configuration_t>(a1),
        std::forward<typename utility::nth_backend<typename F::backend_t, 2>::
                         type::configuration_t>(a2),
        std::forward<typename utility::nth_backend<typename F::backend_t, 3>::
                         type::configuration_t>(a3),
        std::forward<typename utility::nth_backend<typename F::backend_t, 4>::
                         type::configuration_t>(a4)
    );
}

template <
    typename F,
    std::enable_if_t<
        utility::backend_depth<typename F::backend_t>::value == 6,
        bool> = true>
auto make_parameter_pack_for(
    typename utility::nth_backend<typename F::backend_t, 0>::type::
        configuration_t && a0,
    typename utility::nth_backend<typename F::backend_t, 1>::type::
        configuration_t && a1,
    typename utility::nth_backend<typename F::backend_t, 2>::type::
        configuration_t && a2,
    typename utility::nth_backend<typename F::backend_t, 3>::type::
        configuration_t && a3,
    typename utility::nth_backend<typename F::backend_t, 4>::type::
        configuration_t && a4,
    typename utility::nth_backend<typename F::backend_t, 5>::type::
        configuration_t && a5
)
{
    return make_parameter_pack(
        std::forward<typename utility::nth_backend<typename F::backend_t, 0>::
                         type::configuration_t>(a0),
        std::forward<typename utility::nth_backend<typename F::backend_t, 1>::
                         type::configuration_t>(a1),
        std::forward<typename utility::nth_backend<typename F::backend_t, 2>::
                         type::configuration_t>(a2),
        std::forward<typename utility::nth_backend<typename F::backend_t, 3>::
                         type::configuration_t>(a3),
        std::forward<typename utility::nth_backend<typename F::backend_t, 4>::
                         type::configuration_t>(a4),
        std::forward<typename utility::nth_backend<typename F::backend_t, 5>::
                         type::configuration_t>(a5)
    );
}

template <
    typename F,
    std::enable_if_t<
        utility::backend_depth<typename F::backend_t>::value == 7,
        bool> = true>
auto make_parameter_pack_for(
    typename utility::nth_backend<typename F::backend_t, 0>::type::
        configuration_t && a0,
    typename utility::nth_backend<typename F::backend_t, 1>::type::
        configuration_t && a1,
    typename utility::nth_backend<typename F::backend_t, 2>::type::
        configuration_t && a2,
    typename utility::nth_backend<typename F::backend_t, 3>::type::
        configuration_t && a3,
    typename utility::nth_backend<typename F::backend_t, 4>::type::
        configuration_t && a4,
    typename utility::nth_backend<typename F::backend_t, 5>::type::
        configuration_t && a5,
    typename utility::nth_backend<typename F::backend_t, 6>::type::
        configuration_t && a6
)
{
    return make_parameter_pack(
        std::forward<typename utility::nth_backend<typename F::backend_t, 0>::
                         type::configuration_t>(a0),
        std::forward<typename utility::nth_backend<typename F::backend_t, 1>::
                         type::configuration_t>(a1),
        std::forward<typename utility::nth_backend<typename F::backend_t, 2>::
                         type::configuration_t>(a2),
        std::forward<typename utility::nth_backend<typename F::backend_t, 3>::
                         type::configuration_t>(a3),
        std::forward<typename utility::nth_backend<typename F::backend_t, 4>::
                         type::configuration_t>(a4),
        std::forward<typename utility::nth_backend<typename F::backend_t, 5>::
                         type::configuration_t>(a5),
        std::forward<typename utility::nth_backend<typename F::backend_t, 6>::
                         type::configuration_t>(a6)
    );
}

template <
    typename F,
    std::enable_if_t<
        utility::backend_depth<typename F::backend_t>::value == 8,
        bool> = true>
auto make_parameter_pack_for(
    typename utility::nth_backend<typename F::backend_t, 0>::type::
        configuration_t && a0,
    typename utility::nth_backend<typename F::backend_t, 1>::type::
        configuration_t && a1,
    typename utility::nth_backend<typename F::backend_t, 2>::type::
        configuration_t && a2,
    typename utility::nth_backend<typename F::backend_t, 3>::type::
        configuration_t && a3,
    typename utility::nth_backend<typename F::backend_t, 4>::type::
        configuration_t && a4,
    typename utility::nth_backend<typename F::backend_t, 5>::type::
        configuration_t && a5,
    typename utility::nth_backend<typename F::backend_t, 6>::type::
        configuration_t && a6,
    typename utility::nth_backend<typename F::backend_t, 7>::type::
        configuration_t && a7
)
{
    return make_parameter_pack(
        std::forward<typename utility::nth_backend<typename F::backend_t, 0>::
                         type::configuration_t>(a0),
        std::forward<typename utility::nth_backend<typename F::backend_t, 1>::
                         type::configuration_t>(a1),
        std::forward<typename utility::nth_backend<typename F::backend_t, 2>::
                         type::configuration_t>(a2),
        std::forward<typename utility::nth_backend<typename F::backend_t, 3>::
                         type::configuration_t>(a3),
        std::forward<typename utility::nth_backend<typename F::backend_t, 4>::
                         type::configuration_t>(a4),
        std::forward<typename utility::nth_backend<typename F::backend_t, 5>::
                         type::configuration_t>(a5),
        std::forward<typename utility::nth_backend<typename F::backend_t, 6>::
                         type::configuration_t>(a6),
        std::forward<typename utility::nth_backend<typename F::backend_t, 7>::
                         type::configuration_t>(a7)
    );
}

template <
    typename F,
    std::enable_if_t<
        utility::backend_depth<typename F::backend_t>::value == 9,
        bool> = true>
auto make_parameter_pack_for(
    typename utility::nth_backend<typename F::backend_t, 0>::type::
        configuration_t && a0,
    typename utility::nth_backend<typename F::backend_t, 1>::type::
        configuration_t && a1,
    typename utility::nth_backend<typename F::backend_t, 2>::type::
        configuration_t && a2,
    typename utility::nth_backend<typename F::backend_t, 3>::type::
        configuration_t && a3,
    typename utility::nth_backend<typename F::backend_t, 4>::type::
        configuration_t && a4,
    typename utility::nth_backend<typename F::backend_t, 5>::type::
        configuration_t && a5,
    typename utility::nth_backend<typename F::backend_t, 6>::type::
        configuration_t && a6,
    typename utility::nth_backend<typename F::backend_t, 7>::type::
        configuration_t && a7,
    typename utility::nth_backend<typename F::backend_t, 8>::type::
        configuration_t && a8
)
{
    return make_parameter_pack(
        std::forward<typename utility::nth_backend<typename F::backend_t, 0>::
                         type::configuration_t>(a0),
        std::forward<typename utility::nth_backend<typename F::backend_t, 1>::
                         type::configuration_t>(a1),
        std::forward<typename utility::nth_backend<typename F::backend_t, 2>::
                         type::configuration_t>(a2),
        std::forward<typename utility::nth_backend<typename F::backend_t, 3>::
                         type::configuration_t>(a3),
        std::forward<typename utility::nth_backend<typename F::backend_t, 4>::
                         type::configuration_t>(a4),
        std::forward<typename utility::nth_backend<typename F::backend_t, 5>::
                         type::configuration_t>(a5),
        std::forward<typename utility::nth_backend<typename F::backend_t, 6>::
                         type::configuration_t>(a6),
        std::forward<typename utility::nth_backend<typename F::backend_t, 7>::
                         type::configuration_t>(a7),
        std::forward<typename utility::nth_backend<typename F::backend_t, 8>::
                         type::configuration_t>(a8)
    );
}

template <
    typename F,
    std::enable_if_t<
        utility::backend_depth<typename F::backend_t>::value == 10,
        bool> = true>
auto make_parameter_pack_for(
    typename utility::nth_backend<typename F::backend_t, 0>::type::
        configuration_t && a0,
    typename utility::nth_backend<typename F::backend_t, 1>::type::
        configuration_t && a1,
    typename utility::nth_backend<typename F::backend_t, 2>::type::
        configuration_t && a2,
    typename utility::nth_backend<typename F::backend_t, 3>::type::
        configuration_t && a3,
    typename utility::nth_backend<typename F::backend_t, 4>::type::
        configuration_t && a4,
    typename utility::nth_backend<typename F::backend_t, 5>::type::
        configuration_t && a5,
    typename utility::nth_backend<typename F::backend_t, 6>::type::
        configuration_t && a6,
    typename utility::nth_backend<typename F::backend_t, 7>::type::
        configuration_t && a7,
    typename utility::nth_backend<typename F::backend_t, 8>::type::
        configuration_t && a8,
    typename utility::nth_backend<typename F::backend_t, 9>::type::
        configuration_t && a9
)
{
    return make_parameter_pack(
        std::forward<typename utility::nth_backend<typename F::backend_t, 0>::
                         type::configuration_t>(a0),
        std::forward<typename utility::nth_backend<typename F::backend_t, 1>::
                         type::configuration_t>(a1),
        std::forward<typename utility::nth_backend<typename F::backend_t, 2>::
                         type::configuration_t>(a2),
        std::forward<typename utility::nth_backend<typename F::backend_t, 3>::
                         type::configuration_t>(a3),
        std::forward<typename utility::nth_backend<typename F::backend_t, 4>::
                         type::configuration_t>(a4),
        std::forward<typename utility::nth_backend<typename F::backend_t, 5>::
                         type::configuration_t>(a5),
        std::forward<typename utility::nth_backend<typename F::backend_t, 6>::
                         type::configuration_t>(a6),
        std::forward<typename utility::nth_backend<typename F::backend_t, 7>::
                         type::configuration_t>(a7),
        std::forward<typename utility::nth_backend<typename F::backend_t, 8>::
                         type::configuration_t>(a8),
        std::forward<typename utility::nth_backend<typename F::backend_t, 9>::
                         type::configuration_t>(a9)
    );
}
}
