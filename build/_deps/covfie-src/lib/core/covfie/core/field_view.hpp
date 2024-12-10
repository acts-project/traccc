/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <type_traits>

#include <covfie/core/concepts.hpp>
#include <covfie/core/qualifiers.hpp>

namespace covfie {
template <concepts::field_backend _backend>
class field;

template <concepts::field_backend _backend_tc>
class field_view
{
public:
    using backend_t = _backend_tc;
    using storage_t = typename backend_t::non_owning_data_t;
    using output_t = typename backend_t::covariant_output_t::vector_t;
    using coordinate_t = typename backend_t::contravariant_input_t::vector_t;
    using field_t = field<_backend_tc>;

    static_assert(sizeof(storage_t) <= 256, "Storage type is too large.");

    field_view(const field_t & field)
        : m_storage(field.m_backend)
    {
    }

    const storage_t & backend(void)
    {
        return m_storage;
    }

    template <
        typename... Args,
        typename Q = coordinate_t,
        std::enable_if_t<
            std::conjunction_v<std::is_convertible<
                Args,
                typename backend_t::contravariant_input_t::scalar_t>...>,
            bool> = true,
        std::enable_if_t<
            sizeof...(Args) == backend_t::contravariant_input_t::dimensions,
            bool> = true,
        std::enable_if_t<!std::is_scalar_v<Q>, bool> = true>
    COVFIE_DEVICE output_t at(Args... c) const
    {
        return m_storage.at(coordinate_t{
            static_cast<typename backend_t::contravariant_input_t::scalar_t>(c
            )...});
    }

    template <
        typename T,
        std::enable_if_t<std::is_same_v<T, coordinate_t>, bool> = true>
    COVFIE_DEVICE output_t at(T c) const
    {
        return m_storage.at(c);
    }

private:
    storage_t m_storage;
};
}
