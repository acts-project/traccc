/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "utils/axis.hpp"

namespace traccc {

// This object can be iterated to produce the (ordered) set of global indices
// associated with a neighborhood around a certain point on a grid.
//
// The goal is to emulate the effect of enumerating the global indices into
// an std::set (or into an std::vector that gets subsequently sorted), without
// paying the price of dynamic memory allocation in hot magnetic field
// interpolation code.
//
template <size_t DIM>
class global_neighborhood_indices {
   public:
    // You can get the local neighbor indices from
    // grid_helper_impl<DIM>::neighborHoodIndices and the number of bins in
    // each direction from grid_helper_impl<DIM>::getNBins.
    global_neighborhood_indices(
        std::array<neighborhood_indices, DIM>& neighborIndices,
        const std::array<size_t, DIM>& nBinsArray)
        : m_localIndices(neighborIndices) {
        if (DIM == 1)
            return;
        size_t globalStride = 1;
        for (long i = DIM - 2; i >= 0; --i) {
            globalStride *= (nBinsArray[i + 1] + 2);
            m_globalStrides[i] = globalStride;
        }
    }

    class iterator {
       public:
        iterator() = default;

        iterator(
            const global_neighborhood_indices& parent,
            std::array<neighborhood_indices::iterator, DIM>&& localIndicesIter)
            : m_localIndicesIter(std::move(localIndicesIter)),
              m_parent(&parent) {}

        size_t operator*() const {
            size_t globalIndex = *m_localIndicesIter[DIM - 1];
            if (DIM == 1)
                return globalIndex;
            for (size_t i = 0; i < DIM - 1; ++i) {
                globalIndex +=
                    m_parent->m_globalStrides[i] * (*m_localIndicesIter[i]);
            }
            return globalIndex;
        }

        iterator& operator++() {
            const auto& localIndices = m_parent->m_localIndices;

            // Go to the next global index via a lexicographic increment:
            // - Start by incrementing the last local index
            // - If it reaches the end, reset it and increment the previous
            // one...
            for (long i = DIM - 1; i > 0; --i) {
                ++m_localIndicesIter[i];
                if (m_localIndicesIter[i] != localIndices[i].end())
                    return *this;
                m_localIndicesIter[i] = localIndices[i].begin();
            }

            // The first index should stay at the end value when it reaches it,
            // so that we know when we've reached the end of iteration.
            ++m_localIndicesIter[0];
            return *this;
        }

        bool operator==(const iterator& it) {
            // We know when we've reached the end, so we don't need an
            // end-iterator. Sadly, in C++, there has to be one. Therefore, we
            // special-case it heavily so that it's super-efficient to create
            // and compare to.
            if (it.m_parent == nullptr) {
                return m_localIndicesIter[0] ==
                       m_parent->m_localIndices[0].end();
            } else {
                return m_localIndicesIter == it.m_localIndicesIter;
            }
        }

        bool operator!=(const iterator& it) { return !(*this == it); }

       private:
        std::array<neighborhood_indices::iterator, DIM> m_localIndicesIter;
        const global_neighborhood_indices* m_parent = nullptr;
    };

    iterator begin() const {
        std::array<neighborhood_indices::iterator, DIM> localIndicesIter;
        for (size_t i = 0; i < DIM; ++i) {
            localIndicesIter[i] = m_localIndices[i].begin();
        }
        return iterator(*this, std::move(localIndicesIter));
    }

    iterator end() const { return iterator(); }

    // Number of indices that will be produced if this sequence is iterated
    size_t size() const {
        size_t result = m_localIndices[0].size();
        for (size_t i = 1; i < DIM; ++i) {
            result *= m_localIndices[i].size();
        }
        return result;
    }

    // Collect the sequence of indices into an std::vector
    std::vector<size_t> collect() const {
        std::vector<size_t> result;
        result.reserve(this->size());
        for (size_t idx : *this) {
            result.push_back(idx);
        }
        return result;
    }

   private:
    std::array<neighborhood_indices, DIM> m_localIndices;
    std::array<size_t, DIM - 1> m_globalStrides;
};

template <size_t N>
struct grid_helper_impl;

template <size_t N>
struct grid_helper_impl {
    template <class... Axes>
    static void getGlobalBin(
        const std::array<size_t, sizeof...(Axes)>& localBins,
        const std::tuple<Axes...>& axes, size_t& bin, size_t& area) {
        const auto& thisAxis = std::get<N>(axes);
        bin += area * localBins.at(N);
        // make sure to account for under-/overflow bins
        area *= (thisAxis.getNBins() + 2);
        grid_helper_impl<N - 1>::getGlobalBin(localBins, axes, bin, area);
    }

    template <class Point, class... Axes>
    static void getLocalBinIndices(
        const Point& point, const std::tuple<Axes...>& axes,
        std::array<size_t, sizeof...(Axes)>& indices) {
        const auto& thisAxis = std::get<N>(axes);
        indices.at(N) = thisAxis.getBin(point[N]);
        grid_helper_impl<N - 1>::getLocalBinIndices(point, axes, indices);
    }

    template <class... Axes>
    static void getNBins(const std::tuple<Axes...>& axes,
                         std::array<size_t, sizeof...(Axes)>& nBinsArray) {
        // by convention getNBins does not include under-/overflow bins
        nBinsArray[N] = std::get<N>(axes).getNBins();
        grid_helper_impl<N - 1>::getNBins(axes, nBinsArray);
    }

    template <class... Axes>
    static void get_neighborhood_indices(
        const std::array<size_t, sizeof...(Axes)>& localIndices,
        std::pair<size_t, size_t> sizes, const std::tuple<Axes...>& axes,
        std::array<neighborhood_indices, sizeof...(Axes)>& neighborIndices) {
        // ask n-th axis
        size_t locIdx = localIndices.at(N);
        neighborhood_indices locNeighbors =
            std::get<N>(axes).get_neighborhood_indices(locIdx, sizes);
        neighborIndices.at(N) = locNeighbors;

        grid_helper_impl<N - 1>::get_neighborhood_indices(
            localIndices, sizes, axes, neighborIndices);
    }
};

template <>
struct grid_helper_impl<0u> {
    template <class... Axes>
    static void getGlobalBin(
        const std::array<size_t, sizeof...(Axes)>& localBins,
        const std::tuple<Axes...>& /*axes*/, size_t& bin, size_t& area) {
        bin += area * localBins.at(0u);
    }

    template <class Point, class... Axes>
    static void getLocalBinIndices(
        const Point& point, const std::tuple<Axes...>& axes,
        std::array<size_t, sizeof...(Axes)>& indices) {
        const auto& thisAxis = std::get<0u>(axes);
        indices.at(0u) = thisAxis.getBin(point[0u]);
    }

    template <class... Axes>
    static void getNBins(const std::tuple<Axes...>& axes,
                         std::array<size_t, sizeof...(Axes)>& nBinsArray) {
        // by convention getNBins does not include under-/overflow bins
        nBinsArray[0u] = std::get<0u>(axes).getNBins();
    }

    template <class... Axes>
    static void get_neighborhood_indices(
        const std::array<size_t, sizeof...(Axes)>& localIndices,
        std::pair<size_t, size_t> sizes, const std::tuple<Axes...>& axes,
        std::array<neighborhood_indices, sizeof...(Axes)>& neighborIndices) {
        // ask 0-th axis
        size_t locIdx = localIndices.at(0u);
        neighborhood_indices locNeighbors =
            std::get<0u>(axes).get_neighborhood_indices(locIdx, sizes);
        neighborIndices.at(0u) = locNeighbors;
    }
};

struct grid_helper {
    template <class... Axes>
    static size_t getGlobalBin(
        const std::array<size_t, sizeof...(Axes)>& localBins,
        const std::tuple<Axes...>& axes) {
        constexpr size_t MAX = sizeof...(Axes) - 1;
        size_t area = 1;
        size_t bin = 0;

        grid_helper_impl<MAX>::getGlobalBin(localBins, axes, bin, area);

        return bin;
    }

    template <class Point, class... Axes>
    static std::array<size_t, sizeof...(Axes)> getLocalBinIndices(
        const Point& point, const std::tuple<Axes...>& axes) {
        constexpr size_t MAX = sizeof...(Axes) - 1;
        std::array<size_t, sizeof...(Axes)> indices;

        grid_helper_impl<MAX>::getLocalBinIndices(point, axes, indices);

        return indices;
    }

    template <class... Axes>
    static std::array<size_t, sizeof...(Axes)> getNBins(
        const std::tuple<Axes...>& axes) {
        std::array<size_t, sizeof...(Axes)> nBinsArray;
        grid_helper_impl<sizeof...(Axes) - 1>::getNBins(axes, nBinsArray);
        return nBinsArray;
    }

    template <class... Axes>
    static global_neighborhood_indices<sizeof...(Axes)>
    get_neighborhood_indices(
        const std::array<size_t, sizeof...(Axes)>& localIndices,
        std::pair<size_t, size_t> sizes, const std::tuple<Axes...>& axes) {
        constexpr size_t MAX = sizeof...(Axes) - 1;

        // length N array which contains local neighbors based on size par
        std::array<neighborhood_indices, sizeof...(Axes)> neighborIndices;
        // get local bin indices for neighboring bins
        grid_helper_impl<MAX>::get_neighborhood_indices(localIndices, sizes,
                                                        axes, neighborIndices);

        // Query the number of bins
        std::array<size_t, sizeof...(Axes)> nBinsArray = getNBins(axes);

        // Produce iterator of global indices
        return global_neighborhood_indices(neighborIndices, nBinsArray);
    }

    template <class... Axes>
    static global_neighborhood_indices<sizeof...(Axes)>
    get_neighborhood_indices(
        const std::array<size_t, sizeof...(Axes)>& localIndices, size_t size,
        const std::tuple<Axes...>& axes) {
        return get_neighborhood_indices(localIndices,
                                        std::make_pair(size, size), axes);
    }
};

}  // namespace traccc
