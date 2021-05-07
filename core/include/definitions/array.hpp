/**
 * TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <cstddef>

namespace traccc {
    template<typename T, std::size_t N>
    class array {
    public:
        using size_type = std::size_t;

        array(
            void
        ) {
        }

        template<
            typename ... Tp,
            typename = std::enable_if_t<sizeof ...(Tp) == N>,
            typename = std::enable_if_t<(std::is_convertible_v<Tp, T> && ...)>
        >
        array(
            Tp && ... a
        )
            : array(std::index_sequence_for<Tp ...>(), std::forward<Tp>(a) ...)
        {}

        constexpr
        T &
        at(
            size_type i
        ) {
            if (i >= N) {
                throw std::out_of_range();
            }

            return operator[](i);
        }

        constexpr
        const T &
        at(
            size_type i
        ) const {
            if (i >= N) {
                throw std::out_of_range();
            }

            return operator[](i);
        }

        constexpr
        T &
        operator[](
            size_type i
        ) {
            return v[i];
        }

        constexpr
        const T &
        operator[](
            size_type i
        ) const {
            return v[i];
        }

        constexpr
        T &
        front(
            void
        ) {
            return v[0];
        }

        constexpr
        const T &
        front(
            void
        ) const {
            return v[0];
        }

        constexpr
        T &
        back(
            void
        ) {
            return v[N - 1];
        }

        constexpr
        const T &
        back(
            void
        ) const {
            return v[N - 1];
        }

        constexpr
        T *
        data(
            void
        ) {
            return v;
        }

        constexpr
        const T *
        data(
            void
        ) const {
            return v;
        }

    private:
        template<
            size_type ... Is,
            typename ... Tp
        >
        array(
            std::index_sequence<Is ...>,
            Tp && ... a
        ) {
            (static_cast<void>(v[Is] = a), ...);
        }

        T v[N];
    };

    template<typename T, std::size_t N>
    bool
    operator==(
        const array<T, N> & lhs,
        const array<T, N> & rhs
    ) {
        for (typename array<T, N>::size_type i = 0; i < N; ++i) {
            if (lhs[i] != rhs[i]) {
                return false;
            }
        }

        return true;
    }

    template<typename T, std::size_t N>
    bool
    operator!=(
        const array<T, N> & lhs,
        const array<T, N> & rhs
    ) {
        for (typename array<T, N>::size_type i = 0; i < N; ++i) {
            if (lhs[i] == rhs[i]) {
                return false;
            }
        }

        return true;
    }
}
