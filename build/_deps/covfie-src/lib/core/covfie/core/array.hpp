/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cassert>
#include <cstddef>
#include <utility>

#include <covfie/core/qualifiers.hpp>

namespace covfie::array {
template <typename _scalar_t, std::size_t _size>
requires(_size > 0) struct array {
    using scalar_t = _scalar_t;
    using value_type = _scalar_t;
    static constexpr std::size_t dimensions = _size;

    COVFIE_DEVICE array() = default;

    COVFIE_DEVICE array(const scalar_t (&arr)[dimensions])
        requires(dimensions > 1)
        : array(arr, std::make_index_sequence<dimensions>())
    {
    }

    COVFIE_DEVICE array(const scalar_t & val)
        : array(val, std::make_index_sequence<dimensions>())
    {
    }

    template <typename... Ts>
    requires(sizeof...(Ts) == dimensions) COVFIE_DEVICE array(Ts... args)
        : m_data{std::forward<Ts>(args)...}
    {
    }

    COVFIE_DEVICE constexpr scalar_t & at(const std::size_t & n)
    {
        assert(n < dimensions);

        return m_data[n];
    }

    COVFIE_DEVICE constexpr const scalar_t & at(const std::size_t & n) const
    {
        assert(n < dimensions);

        return m_data[n];
    }

    COVFIE_DEVICE constexpr scalar_t & operator[](const std::size_t & n)
    {
        assert(n < dimensions);

        return m_data[n];
    }

    COVFIE_DEVICE constexpr const scalar_t & operator[](const std::size_t & n
    ) const
    {
        assert(n < dimensions);

        return m_data[n];
    }

    COVFIE_DEVICE constexpr std::size_t size() const
    {
        return dimensions;
    }

    COVFIE_DEVICE constexpr scalar_t * begin()
    {
        return m_data + 0;
    }

    COVFIE_DEVICE constexpr const scalar_t * begin() const
    {
        return m_data + 0;
    }

    COVFIE_DEVICE constexpr const scalar_t * cbegin() const
    {
        return m_data + 0;
    }

    COVFIE_DEVICE constexpr scalar_t * end()
    {
        return m_data + dimensions;
    }

    COVFIE_DEVICE constexpr const scalar_t * end() const
    {
        return m_data + dimensions;
    }

    COVFIE_DEVICE constexpr const scalar_t * cend() const
    {
        return m_data + dimensions;
    }

private:
    template <std::size_t... Is>
    COVFIE_DEVICE
    array(const scalar_t (&arr)[dimensions], std::index_sequence<Is...>)
        : m_data{arr[Is]...}
    {
    }

    template <std::size_t... Is>
    COVFIE_DEVICE array(const scalar_t & val, std::index_sequence<Is...>)
        : m_data{((void)Is, val)...}
    {
    }

    scalar_t m_data[dimensions];
};
}
