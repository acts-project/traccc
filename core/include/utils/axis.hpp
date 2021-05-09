/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc{

class neighborhood_indices{
public:
    neighborhood_indices() = default;
    
    neighborhood_indices(size_t begin, size_t end)
	: m_begin1(begin), m_end1(end), m_begin2(end), m_end2(end) {}
    
    neighborhood_indices(size_t begin1, size_t end1, size_t begin2, size_t end2)
	: m_begin1(begin1), m_end1(end1), m_begin2(begin2), m_end2(end2) {}

    class iterator{
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
    
// Defined only for bounded axis
template <AxisBoundaryType bdt>
class axis {
public:    
    axis(scalar xmin, scalar xmax, size_t n_Bins):
	m_min(xmin),
	m_max(xmax),
	m_width((xmax-xmin)/n_Bins),
	m_bins(n_Bins){}

    template <AxisBoundaryType T = bdt,
	      std::enable_if_t<T == AxisBoundaryType::Bound, int> = 0>
    size_t wrapBin(int bin) const {
	return std::max(std::min(bin, static_cast<int>(getNBins())), 1);
    }

    template <AxisBoundaryType T = bdt,
	      std::enable_if_t<T == AxisBoundaryType::Closed, int> = 0>
    size_t wrapBin(int bin) const {
	const int w = getNBins();
	return 1 + (w + ((bin - 1) % w)) % w;
	// return int(bin<1)*w - int(bin>w)*w + bin;
    }    
    
    scalar getMin() const { return m_min; }

    scalar getBinWidth(size_t /*bin*/ = 0) const { return m_width; }
    
    size_t getBin(scalar x) const {
	return wrapBin(std::floor((x - getMin()) / getBinWidth()) + 1);
    }
    
    size_t getNBins() const { return m_bins; }

    
private:
    scalar m_min;
    scalar m_max;
    scalar m_width;
    size_t m_bins;	
};

}
