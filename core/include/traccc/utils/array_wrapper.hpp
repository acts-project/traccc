/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <traccc/utils/functor.hpp>
#include <type_traits>
#include <utility>
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/memory/unique_ptr.hpp>

namespace traccc::cuda {

template <typename... Ts>
struct pod {};

template <>
struct pod<> {};

template <typename T, typename... Ts>
struct pod<T, Ts...> {
    T v;
    pod<Ts...> r;
};

template <std::size_t I, typename... Ts>
constexpr auto& pod_get(pod<Ts...>& p) {
    if constexpr (I == 0) {
        return p.v;
    } else {
        return pod_get<I - 1>(p.r);
    }
}

template <template <typename...> typename F,
          template <template <typename> typename> typename T>
struct array_wrapper {
    struct owner {
        owner(vecmem::memory_resource& mr, std::size_t n) : data(mr, n) {}

        typename details::functor::reapply<
            F, typename T<details::functor::identity>::tuple_t>::type::owner
            data;
    };

    struct handle {
        using handle_t = typename details::functor::reapply<
            F, typename T<details::functor::identity>::tuple_t>::type::handle;

        handle(const owner& o) : data(o.data) {}

        __host__ __device__ std::size_t size() const { return data.size(); }

        template <std::size_t I>
        __host__ __device__ auto& get(std::size_t i) {
            return data.get<I>(i);
        }

        template <std::size_t I>
        __host__ __device__ auto get(std::size_t i) const {
            return data.get<I>(i);
        }

        template <std::size_t I>
        __host__ __device__ auto get_identity(std::size_t i) const {
            return data.get<I>(i);
        }

        template <std::size_t I>
        __host__ __device__ auto& get_reference(std::size_t i) {
            return data.get<I>(i);
        }

        template <std::size_t... Ns>
        constexpr __host__ __device__ T<details::functor::identity>
        _construct_helper_identity(std::index_sequence<Ns...>,
                                   std::size_t i) const {
            return T<details::functor::identity>{get_identity<Ns>(i)...};
        }

        template <std::size_t... Ns>
        constexpr __host__ __device__ T<details::functor::reference>
        _construct_helper_reference(std::index_sequence<Ns...>, std::size_t i) {
            return T<details::functor::reference>{get_reference<Ns>(i)...};
        }

        __host__ __device__ T<details::functor::identity> operator[](
            std::size_t i) const {
            return _construct_helper_identity(
                std::make_index_sequence<std::tuple_size_v<
                    typename T<details::functor::identity>::tuple_t>>(),
                i);
        }

        __host__ __device__ T<details::functor::reference> operator[](
            std::size_t i) {
            return _construct_helper_reference(
                std::make_index_sequence<std::tuple_size_v<
                    typename T<details::functor::identity>::tuple_t>>(),
                i);
        }

        handle_t data;
    };
};

template <std::size_t... Ns, typename... Ts>
std::tuple<Ts*...> _get_ptrs(
    std::index_sequence<Ns...>,
    const std::tuple<vecmem::unique_alloc_ptr<Ts[]>...>& o) {
    return {std::get<Ns>(o).get()...};
}

template <typename... Ts>
struct soa {
    struct owner {
        owner(vecmem::memory_resource& mr, std::size_t n)
            : _size(n), _ptrs{vecmem::make_unique_alloc<Ts[]>(mr, n)...} {}

        std::size_t size() const { return _size; }

        const std::tuple<vecmem::unique_alloc_ptr<Ts[]>...>& pointers() const {
            return _ptrs;
        }

        private:
        std::size_t _size;
        std::tuple<vecmem::unique_alloc_ptr<Ts[]>...> _ptrs;
    };

    struct handle {
        private:
        using tuple_t = std::tuple<Ts*...>;

        public:
        handle(const owner& o)
            : _size(o.size()),
              _ptrs(_get_ptrs(std::make_index_sequence<sizeof...(Ts)>(),
                              o.pointers())) {}

        __host__ __device__ std::size_t size() const { return _size; }

        template <std::size_t I>
        __host__ __device__ auto& get(std::size_t i) {
            return std::get<I>(_ptrs)[i];
        }

        template <std::size_t I>
        __host__ __device__ const auto& get(std::size_t i) const {
            return std::get<I>(_ptrs)[i];
        }

        private:
        std::size_t _size;
        std::tuple<Ts*...> _ptrs;
    };
};

template <typename... Ts>
struct aos {
    struct owner {
        owner(vecmem::memory_resource& mr, std::size_t n)
            : _size(n), _ptr{vecmem::make_unique_alloc<pod<Ts...>[]>(mr, n)} {}

        std::size_t size() const { return _size; }

        const vecmem::unique_alloc_ptr<pod<Ts...>[]>& pointer() const {
            return _ptr;
        }

        private : std::size_t _size;
        vecmem::unique_alloc_ptr<pod<Ts...>[]> _ptr;
    };

    struct handle {
        private:
        using tuple_t = pod<Ts...>;

        public:
        handle(const owner& o) : _size(o.size()), _ptr(o.pointer().get()) {}

        __host__ __device__ std::size_t size() const { return _size; }

        template <std::size_t I>
        __host__ __device__ auto& get(std::size_t i) {
            return pod_get<I>(_ptr[i]);
        }

        template <std::size_t I>
        __host__ __device__ const auto& get(std::size_t i) const {
            return pod_get<I>(_ptr[i]);
        }

        private:
        std::size_t _size;
        tuple_t* _ptr;
    };
};
}  // namespace traccc::cuda
