/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc {

// This object can be iterated to produce up to two sequences of integer
// indices, corresponding to the half-open integer ranges [begin1, end1[ and
// [begin2, end2[.
//
// The goal is to emulate the effect of enumerating a range of neighbor
// indices on an axis (which may go out of bounds and wrap around since we
// have AxisBoundaryType::Closed), inserting them into an std::vector, and
// discarding duplicates, without paying the price of duplicate removal
// and dynamic memory allocation in hot magnetic field interpolation code.
//
class neighborhood_indices {
   public:
    neighborhood_indices() = default;

    neighborhood_indices(size_t begin, size_t end)
        : m_begin1(begin), m_end1(end), m_begin2(end), m_end2(end) {}

    neighborhood_indices(size_t begin1, size_t end1, size_t begin2, size_t end2)
        : m_begin1(begin1), m_end1(end1), m_begin2(begin2), m_end2(end2) {}

    class iterator {
       public:
        iterator() = default;
        // Specialized constructor for end() iterator
        iterator(size_t current) : m_current(current), m_wrapped(true) {}

        iterator(size_t begin1, size_t end1, size_t begin2)
            : m_current(begin1),
              m_end1(end1),
              m_begin2(begin2),
              m_wrapped(begin1 == begin2) {}

        size_t operator*() const { return m_current; }

        iterator& operator++() {
            ++m_current;
            if (m_current == m_end1) {
                m_current = m_begin2;
                m_wrapped = true;
            }
            return *this;
        }

        bool operator==(const iterator& it) const {
            return (m_current == it.m_current) && (m_wrapped == it.m_wrapped);
        }

        bool operator!=(const iterator& it) const { return !(*this == it); }

       private:
        size_t m_current, m_end1, m_begin2;
        bool m_wrapped;
    };

    iterator begin() const { return iterator(m_begin1, m_end1, m_begin2); }

    iterator end() const { return iterator(m_end2); }

    // Number of indices that will be produced if this sequence is iterated
    size_t size() const { return (m_end1 - m_begin1) + (m_end2 - m_begin2); }

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
    size_t m_begin1 = 0, m_end1 = 0, m_begin2 = 0, m_end2 = 0;
};

enum class AxisBoundaryType { Open, Bound, Closed };

template <AxisBoundaryType bdt>
class axis {
   public:
    axis(scalar xmin, scalar xmax, size_t n_Bins)
        : m_min(xmin),
          m_max(xmax),
          m_width((xmax - xmin) / n_Bins),
          m_bins(n_Bins) {}

    /// @brief Converts bin index into a valid one for this axis.
    ///
    /// @note Bound: bin index is clamped to [1, nBins]
    ///
    /// @param [in] bin The bin to wrap
    /// @return valid bin index
    template <AxisBoundaryType T = bdt,
              std::enable_if_t<T == AxisBoundaryType::Bound, int> = 0>
    size_t wrapBin(int bin) const {
        return std::max(std::min(bin, static_cast<int>(getNBins())), 1);
    }

    /// @brief Converts bin index into a valid one for this axis.
    ///
    /// @note Closed: bin index wraps around to other side
    ///
    /// @param [in] bin The bin to wrap
    /// @return valid bin index
    template <AxisBoundaryType T = bdt,
              std::enable_if_t<T == AxisBoundaryType::Closed, int> = 0>
    size_t wrapBin(int bin) const {
        const int w = getNBins();
        return 1 + (w + ((bin - 1) % w)) % w;
        // return int(bin<1)*w - int(bin>w)*w + bin;
    }

    /// @brief Get #size bins which neighbor the one given
    ///
    /// This is the version for Bound
    ///
    /// @param [in] idx requested bin index
    /// @param [in] sizes how many neighboring bins (up/down)
    /// @return Set of neighboring bin indices (global)
    /// @note Bound varies given bin and allows 1 and NBins (regular bins)
    ///       as neighbors
    template <AxisBoundaryType T = bdt,
              std::enable_if_t<T == AxisBoundaryType::Bound, int> = 0>
    neighborhood_indices get_neighborhood_indices(
        size_t idx, std::pair<size_t, size_t> sizes = {1, 1}) const {
        if (idx <= 0 || idx >= (getNBins() + 1)) {
            return neighborhood_indices();
        }
        constexpr int min = 1;
        const int max = getNBins();
        const int itmin = std::max(min, int(idx - sizes.first));
        const int itmax = std::min(max, int(idx + sizes.second));
        return neighborhood_indices(itmin, itmax + 1);
    }

    /// @brief Get #size bins which neighbor the one given
    ///
    /// This is the version for Closed
    ///
    /// @param [in] idx requested bin index
    /// @param [in] sizes how many neighboring bins (up/down)
    /// @return Set of neighboring bin indices (global)
    /// @note Closed varies given bin and allows bins on the opposite
    ///       side of the axis as neighbors. (excludes underflow / overflow)
    template <AxisBoundaryType T = bdt,
              std::enable_if_t<T == AxisBoundaryType::Closed, int> = 0>
    neighborhood_indices get_neighborhood_indices(
        size_t idx, std::pair<size_t, size_t> sizes = {1, 1}) const {
        // Handle invalid indices
        if (idx <= 0 || idx >= (getNBins() + 1)) {
            return neighborhood_indices();
        }

        // Handle corner case where user requests more neighbours than the
        // number of bins on the axis. We do not want to ActsScalar-count bins
        // in that case.
        sizes.first %= getNBins();
        sizes.second %= getNBins();
        if (sizes.first + sizes.second + 1 > getNBins()) {
            sizes.second -= (sizes.first + sizes.second + 1) - getNBins();
        }

        // If the entire index range is not covered, we must wrap the range of
        // targeted neighbor indices into the range of valid bin indices. This
        // may split the range of neighbor indices in two parts:
        //
        // Before wraparound - [        XXXXX]XXX
        // After wraparound  - [ XXXX   XXXX ]
        //
        const int itmin = idx - sizes.first;
        const int itmax = idx + sizes.second;
        const size_t itfirst = wrapBin(itmin);
        const size_t itlast = wrapBin(itmax);
        if (itfirst <= itlast) {
            return neighborhood_indices(itfirst, itlast + 1);
        } else {
            return neighborhood_indices(itfirst, getNBins() + 1, 1, itlast + 1);
        }
    }

    /// @brief get minimum of binning range
    ///
    /// @return minimum of binning range
    scalar getMin() const { return m_min; }

    /// @brief get bin width
    ///
    /// @param  [in] bin index of bin
    /// @return width of given bin
    ///
    /// @pre @c bin must be a valid bin index (excluding under-/overflow bins),
    ///      i.e. \f$1 \le \text{bin} \le \text{nBins}\f$
    scalar getBinWidth(size_t /*bin*/ = 0) const { return m_width; }

    /// @brief get corresponding bin index for given coordinate
    ///
    /// @param  [in] x input coordinate
    /// @return index of bin containing the given value
    ///
    /// @note Bin intervals are defined with closed lower bounds and open upper
    ///       bounds, that is \f$l <= x < u\f$ if the value @c x lies within a
    ///       bin with lower bound @c l and upper bound @c u.
    /// @note Bin indices start at @c 1. The underflow bin has the index @c 0
    ///       while the index <tt>nBins + 1</tt> indicates the overflow bin .
    size_t getBin(scalar x) const {
        return wrapBin(std::floor((x - getMin()) / getBinWidth()) + 1);
    }

    /// @brief get total number of bins
    ///
    /// @return total number of bins (excluding under-/overflow bins)
    size_t getNBins() const { return m_bins; }

   private:
    scalar m_min;
    scalar m_max;
    scalar m_width;
    size_t m_bins;
};

}  // namespace traccc
