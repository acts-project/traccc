/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <iostream>

#include <covfie/core/concepts.hpp>
#include <covfie/core/field_view.hpp>
#include <covfie/core/parameter_pack.hpp>
#include <covfie/core/utility/binary_io.hpp>

namespace covfie {
template <concepts::field_backend _backend_t>
class field
{
public:
    using backend_t = _backend_t;
    using view_t = field_view<backend_t>;
    using storage_t = typename backend_t::owning_data_t;
    using output_t = typename backend_t::covariant_output_t::vector_t;
    using coordinate_t = typename backend_t::contravariant_input_t::vector_t;

    static constexpr uint32_t IO_MAGIC_HEADER = 0xAB000000;

    field() = default;
    field(const field &) = default;
    field(field &&) = default;

    template <concepts::field_backend other_backend>
    explicit field(field<other_backend> && other)
        : m_backend(std::move(other.m_backend))
    {
    }

    template <concepts::field_backend other_backend>
    explicit field(const field<other_backend> & other)
        : m_backend(other.m_backend)
    {
    }

    template <typename... Args>
    explicit field(parameter_pack<Args...> && args)
        : m_backend(std::forward<parameter_pack<Args...>>(args))
    {
    }

    explicit field(std::istream & fs)
        : m_backend(decltype(m_backend
          )::read_binary(utility::read_io_header(fs, IO_MAGIC_HEADER)))
    {
        utility::read_io_footer(fs, IO_MAGIC_HEADER);
    }

    field & operator=(const field &) = default;

    field & operator=(field &&) = default;

    const storage_t & backend(void) const
    {
        return m_backend;
    }

    void dump(std::ostream & fs) const
    {
        utility::write_io_header(fs, IO_MAGIC_HEADER);
        backend_t::owning_data_t::write_binary(fs, m_backend);
        utility::write_io_footer(fs, IO_MAGIC_HEADER);
    }

private:
    storage_t m_backend;

    friend class field_view<_backend_t>;

    template <concepts::field_backend other_backend>
    friend class field;
};
}
