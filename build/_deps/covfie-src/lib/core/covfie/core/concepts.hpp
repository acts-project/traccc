/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <concepts>
#include <iostream>
#include <optional>

#include <covfie/core/definitions.hpp>
#include <covfie/core/parameter_pack.hpp>
namespace covfie::concepts {
template <typename T>
concept is_initial = T::is_initial == true;

template <typename T>
concept is_constructible_from_config_and_backend = requires(
    const typename T::configuration_t & c,
    typename T::backend_t::owning_data_t & b
)
{
    {typename T::owning_data_t(c, std::move(b))};
};

template <typename T>
concept is_constructible_from_parameter_pack_len_gt1 =
    requires(parameter_pack<typename T::configuration_t, std::monostate> && p)
{
    {typename T::owning_data_t(std::move(p))};
};

template <typename T>
concept is_constructible_from_parameter_pack_len_eq1 =
    requires(parameter_pack<typename T::configuration_t> && p)
{
    {typename T::owning_data_t(std::move(p))};
};

template <typename T>
concept is_constructible_from_parameter_pack_self =
    requires(parameter_pack<typename T::owning_data_t> && p)
{
    {typename T::owning_data_t(std::move(p))};
};

template <typename T>
concept is_constructible_from_self_rvalue =
    requires(typename T::owning_data_t && p)
{
    {typename T::owning_data_t(std::move(p))};
};

template <typename T>
concept field_backend = requires
{
    /*
     * Every backend should have an alias to itself. Not great, but sometimes
     * C++ necessitates these things.
     */
    typename T::this_t;

    /*
     * Check whether the backend has the required type definitions, which are
     * the input and output types of the contravariant and covariant parts of
     * the layer.
     */
    typename T::contravariant_input_t;
    typename T::covariant_output_t;

    /*
     * Confirm that the backend has both an owning and a non-owning data type.
     */
    typename T::owning_data_t;
    typename T::non_owning_data_t;

    /*
     * Every bit of data, owning and non-owning, must make available which
     * backend it belongs to.
     */
    typename T::owning_data_t::parent_t;
    typename T::non_owning_data_t::parent_t;

    /*
     * The non-owning data which we use on the GPU must be extremely simple,
     * because we cannot model destructors and such there properly.
     */
    requires std::is_trivially_destructible_v<typename T::non_owning_data_t>;
    requires
        std::is_trivially_copy_constructible_v<typename T::non_owning_data_t>;
    requires
        std::is_trivially_move_constructible_v<typename T::non_owning_data_t>;

    /*
     * Backends must declare whether they are initial or not; this must be done
     * through a static constexpr boolean.
     */
    T::is_initial;
    requires std::is_same_v<bool, std::decay_t<decltype(T::is_initial)>>;

    /*
     * Types must declare whether they are configuration-constructible, which
     * is to say they can be constructed from a configuration structure. Note
     * that the configuration is stricly for the transformer, and not for its
     * children.
     */
    typename T::configuration_t;

    requires std::is_trivially_destructible_v<typename T::configuration_t>;
    requires
        std::is_trivially_copy_constructible_v<typename T::configuration_t>;
    requires
        std::is_trivially_move_constructible_v<typename T::configuration_t>;

    requires requires(const typename T::owning_data_t & o)
    {
        {
            o.get_configuration()
        } -> std::same_as<typename T::configuration_t>;
    };

    /*
     * Check for additional constructability from parameter packs.
     */
    requires is_initial<T> || is_constructible_from_parameter_pack_len_gt1<T>;
    requires !is_initial<T> || is_constructible_from_parameter_pack_len_eq1<T>;
    requires is_constructible_from_parameter_pack_self<T>;

    /*
     * Check whether the field is constructible from itself.
     */
    requires is_constructible_from_self_rvalue<T>;

    /*
     * Check whether the owning data type can be read from a file.
     */
    requires requires(std::istream & fs)
    {
        {
            T::owning_data_t::read_binary(fs)
        } -> std::same_as<typename T::owning_data_t>;
    };

    requires requires(std::ostream & fs, const typename T::owning_data_t & o)
    {
        {
            T::owning_data_t::write_binary(fs, o)
        } -> std::same_as<void>;
    };

    requires is_initial<T> || is_constructible_from_config_and_backend<T>;

    {typename T::owning_data_t()};

    requires requires(typename T::owning_data_t & d)
    {
        requires T::is_initial || requires
        {
            typename T::backend_t;
            {
                d.get_backend()
            } -> std::same_as<typename T::backend_t::owning_data_t &>;
        };

        {typename T::owning_data_t(d)};
        {typename T::owning_data_t(std::move(d))};

        requires requires(typename T::owning_data_t & e)
        {
            {d = e};
            {d = std::move(e)};
        };
    };

    requires requires(const typename T::owning_data_t & d)
    {
        /*
         * Check whether a non-owning data type can be constructed from the
         * owning variant.
         */
        {typename T::non_owning_data_t(d)};

        /*
         * Any non-initial backend must allow an accessor method to access the
         * non-owning data type of its innards.
         */
        requires T::is_initial || requires
        {
            typename T::backend_t;
            {
                d.get_backend()
            } -> std::same_as<const typename T::backend_t::owning_data_t &>;
        };
    };

    requires requires(typename T::non_owning_data_t & d)
    {
        /*
         * Just like how owning data types can spit out references to their
         * children, so can non-owning data types.
         */
        requires T::is_initial || requires
        {
            typename T::backend_t;
            {
                d.get_backend()
            } -> std::same_as<typename T::backend_t::non_owning_data_t &>;
        };
    };

    /*
     * Check whether a non-owning object allows us to look up the magnetic
     * field, and whether that operation gives the correct result.
     */
    requires requires(const typename T::non_owning_data_t & d)
    {
        {
            d.at(std::declval<typename T::contravariant_input_t::vector_t>())
        } -> std::same_as<typename T::covariant_output_t::vector_t>;

        /*
         * Constant version of the requirement that non-owning data can be
         * introspected.
         */
        requires T::is_initial || requires
        {
            typename T::backend_t;
            {
                d.get_backend()
            } -> std::same_as<const typename T::backend_t::non_owning_data_t &>;
        };
    };
};

template <typename T>
concept vector_descriptor = true;
}
