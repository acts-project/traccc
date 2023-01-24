/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <type_traits>
#include <variant>

#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/concept.hpp"
#include "traccc/utils/tuple.hpp"
#include "traccc/utils/type_traits.hpp"

namespace traccc::cuda {
#ifdef TRACCC_HAVE_CONCEPTS
template <typename T>
concept graph_descriptor_c = requires {
    typename T::result_type;
    typename T::config_type;
    typename T::argument_type;
};

template <typename T>
concept noninitial_graph_descriptor_c = graph_descriptor_c<T>and requires {
    requires requires(cudaGraph_t g, cudaGraphNode_t n,
                      typename T::config_type c, typename T::argument_type a) {
        { T::append_graph(g, n, c, a) }
        ->std::same_as<std::tuple<cudaGraphNode_t, typename T::result_type>>;
    };
};

template <typename T>
concept initial_graph_descriptor_c = graph_descriptor_c<T>and requires {
    requires requires(typename T::config_type c, typename T::argument_type a) {
        { T::create_graph(c, a) }
        ->std::same_as<
            std::tuple<cudaGraph_t, cudaGraphNode_t, typename T::result_type>>;
    };
};
#endif

namespace details {
template <CONSTRAINT(initial_graph_descriptor_c) G1,
          CONSTRAINT(noninitial_graph_descriptor_c) G2>
class compose_graphs2i {
    public:
    using result_type = typename G2::result_type;
    using config_type = decltype(std::tuple_cat(
        std::declval<std::tuple<typename G1::config_type>>(),
        std::declval<std::conditional_t<
            traccc::details::is_tuple<typename G2::config_type>::value,
            typename G2::config_type,
            std::tuple<typename G2::config_type>>>()));
    using argument_type = typename G1::argument_type;

    static std::tuple<cudaGraph_t, cudaGraphNode_t, result_type> create_graph(
        config_type c, argument_type a0) {
        cudaGraph_t g;
        cudaGraphNode_t n;
        typename G1::result_type a1;

        std::tie(g, n, a1) =
            G1::create_graph(traccc::details::tuple_head(c), a0);
        return std::tuple_cat(
            std::tuple<cudaGraph_t>{g},
            G2::append_graph(g, n, traccc::details::tuple_tail(c), a1));

        if constexpr (traccc::details::is_tuple<
                          typename G2::config_type>::value) {
            return std::tuple_cat(
                std::tuple<cudaGraph_t>{g},
                G2::append_graph(g, n, traccc::details::tuple_tail(c), a1));
        } else {
            return std::tuple_cat(
                std::tuple<cudaGraph_t>{g},
                G2::append_graph(
                    g, n, std::get<0>(traccc::details::tuple_tail(c)), a1));
        }
    }
};

template <CONSTRAINT(noninitial_graph_descriptor_c) G1,
          CONSTRAINT(noninitial_graph_descriptor_c) G2>
class compose_graphs2 {
    public:
    using result_type = typename G2::result_type;
    using config_type = decltype(std::tuple_cat(
        std::declval<std::tuple<typename G1::config_type>>(),
        std::declval<std::conditional_t<
            traccc::details::is_tuple<typename G2::config_type>::value,
            typename G2::config_type,
            std::tuple<typename G2::config_type>>>()));
    using argument_type = typename G1::argument_type;

    static std::tuple<cudaGraphNode_t, result_type> append_graph(
        cudaGraph_t g, cudaGraphNode_t n0, config_type c, argument_type a0) {

        cudaGraphNode_t n1;
        typename G1::result_type a1;

        std::tie(n1, a1) =
            G1::append_graph(g, n0, traccc::details::tuple_head(c), a0);

        if constexpr (traccc::details::is_tuple<
                          typename G2::config_type>::value) {
            return G2::append_graph(g, n1, traccc::details::tuple_tail(c), a1);
        } else {
            return G2::append_graph(
                g, n1, std::get<0>(traccc::details::tuple_tail(c)), a1);
        }
    }
};

template <CONSTRAINT(noninitial_graph_descriptor_c)...>
struct compose_graphs {};

template <CONSTRAINT(noninitial_graph_descriptor_c) G>
struct compose_graphs<G> {
    using type = G;
};

template <CONSTRAINT(noninitial_graph_descriptor_c) G1,
          CONSTRAINT(noninitial_graph_descriptor_c)... Gs>
struct compose_graphs<G1, Gs...> {
    static_assert(
        !traccc::details::is_tuple<typename G1::config_type>::value &&
            ((!traccc::details::is_tuple<typename Gs::config_type>::value) &&
             ...),
        "Graphs in composition must not have tuple configuration type.");

    using type = compose_graphs2<G1, typename compose_graphs<Gs...>::type>;
};

template <CONSTRAINT(initial_graph_descriptor_c) G1,
          CONSTRAINT(noninitial_graph_descriptor_c)... Gs>
class compose_graphs_initial {
    static_assert(
        !traccc::details::is_tuple<typename G1::config_type>::value &&
            ((!traccc::details::is_tuple<typename Gs::config_type>::value) &&
             ...),
        "Graphs in composition must not have tuple configuration type.");

    public:
    using type = std::conditional_t<
        sizeof...(Gs) == 0, G1,
        compose_graphs2i<G1, typename compose_graphs<Gs...>::type>>;
};
}  // namespace details

template <CONSTRAINT(noninitial_graph_descriptor_c) G1,
          CONSTRAINT(noninitial_graph_descriptor_c)... Gs>
using compose_graphs = typename details::compose_graphs<G1, Gs...>::type;

template <CONSTRAINT(initial_graph_descriptor_c) G1,
          CONSTRAINT(noninitial_graph_descriptor_c)... Gs>
using compose_graphs_initial =
    typename details::compose_graphs_initial<G1, Gs...>::type;

}  // namespace traccc::cuda
