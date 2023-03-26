/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::details::functor {
template <typename T>
using identity = T;

template <typename T>
using reference = T&;

template <typename T>
using const_reference = const T&;

template <template <typename...> typename F, typename T>
struct reapply {};

template <template <typename...> typename F1,
          template <typename...> typename F2, typename... Ts>
struct reapply<F1, F2<Ts...>> {
    using type = F1<Ts...>;
};
}  // namespace traccc::details::functor
