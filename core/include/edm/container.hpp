/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <type_traits>

// VecMem include(s).
#include <vecmem/containers/data/jagged_vector_buffer.hpp>
#include <vecmem/containers/data/vector_buffer.hpp>
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/jagged_device_vector.hpp>
#include <vecmem/containers/jagged_vector.hpp>
#include <vecmem/containers/vector.hpp>

#include "definitions/qualifiers.hpp"

namespace traccc {
/**
 * @brief View class for an element in a header-vector container.
 *
 * In order to enforce certain invariants on the @c container header-vector
 * type, we access elements (which, of course, are product types of a header
 * and a vector) through this wrapper class. This class provides
 * low-overhead access to the struct-of-arrays container class to emulate an
 * array-of-structs architecture.
 *
 * @tparam header_t The type of the header object.
 * @tparam vector_t The fully qualified vector type.
 */
template <typename header_t, typename vector_t>
class container_element {
    public:
    /**
     * @brief Construct a new container element view.
     *
     * This constructor is extremely trivial, as it simply takes a reference
     * to a header, a reference to a vector, and saves them in this object's
     * internal state.
     *
     * @param[in] h The header object reference.
     * @param[in] v The vector object reference.
     */
    TRACCC_HOST_DEVICE
    container_element(header_t& h, vector_t& v) : header(h), items(v) {}

    header_t& header;
    vector_t& items;
};

/// Container describing objects in a given event
///
/// This is the generic container of the code, holding all relevant
/// information about objcts in a given event.
///
/// It can be instantiated with different vector types, to be able to use
/// the same container type in both host and device code.
///
/// It also can be instantiated with different edm types represented by
/// header and item type.
template <typename header_t, typename item_t,
          template <typename> class vector_t,
          template <typename> class jagged_vector_t,
          template <typename, typename> class pair_t = std::pair>
class container {
    public:
    /// @name Type definitions
    /// @{

    /// Vector type used by the container
    template <typename T>
    using vector_type = vector_t<T>;

    /// Jagged vector type used by the container
    template <typename T>
    using jagged_vector_type = jagged_vector_t<T>;

    /// The header vector type
    using header_vector = vector_type<header_t>;

    /// The item vector type
    using item_vector = jagged_vector_type<item_t>;

    /**
     * @brief The size type of this container, which is the type by which
     * its elements are indexed.
     */
    using size_type = typename header_vector::size_type;

    /// The element link type
    using link_type = pair_t<typename header_vector::size_type,
                             typename item_vector::size_type>;

    /// @}

    /**
     * @brief The type name of the element view which is returned by various
     * methods in this class.
     */
    using element_view =
        container_element<header_t, typename item_vector::value_type>;

    /**
     * @brief The type name of the constant element view which is returned
     * by various methods in this class.
     */
    using const_element_view =
        container_element<const header_t,
                          const typename item_vector::value_type>;

    /**
     * We need to assert that the header vector and the outer layer of the
     * jagged vector have the same size type, so they can be indexed using
     * the same type.
     */
    static_assert(
        std::is_convertible<typename header_vector::size_type,
                            typename item_vector::size_type>::value,
        "Size type for container header and item vectors must be the same.");

    /**
     * @brief Standard two-argument constructor.
     *
     * To enforce the invariant that both vectors must be the same size, we
     * check this in the constructor. This is also checked in release
     * builds.
     */
    TRACCC_HOST
    container(header_vector&& hv, item_vector&& iv) : headers(hv), items(iv) {
        if (headers.size() != items.size()) {
            throw std::logic_error("Header and item length not equal.");
        }
    }

    template <typename header_vector_tp, typename item_vector_tp,
              typename = std::enable_if<
                  std::is_same<header_vector_tp, vector_t<header_t>>::value>,
              typename = std::enable_if<
                  std::is_same<item_vector_tp, jagged_vector_t<item_t>>::value>>
    TRACCC_HOST_DEVICE container(header_vector_tp&& hv, item_vector_tp&& iv)
        : headers(hv), items(iv) {
#if defined(__CUDACC__) || defined(__SYCL_DEVICE_ONLY__)
        assert(headers.size() == items.size());
#else
        if (headers.size() != items.size()) {
            throw std::logic_error("Header and item length not equal.");
        }
#endif
    }

    /**
     * @brief Constructor with vector size and memory resource .
     */
    template <typename size_type>
    TRACCC_HOST explicit container(size_type size, vecmem::memory_resource* mr)
        : headers(size, mr), items(size, mr) {}

    /**
     * @brief Constructor with memory resource .
     */

    TRACCC_HOST explicit container(vecmem::memory_resource* mr)
        : headers(mr), items(mr) {}

    /**
     * @brief Default Constructor
     */
    container() = default;

    /**
     * @brief Bounds-checking mutable element accessor.
     */
    TRACCC_HOST
    element_view at(size_type i) {
        if (i >= size()) {
            throw std::out_of_range("Index out of range.");
        }
        return operator[](i);
    }

    /**
     * @brief Bounds-checking immutable element accessor.
     */
    TRACCC_HOST
    const_element_view at(size_type i) const {
        if (i >= size()) {
            throw std::out_of_range("Index out of range.");
        }
        return operator[](i);
    }

    /**
     * @brief Bounds-checking mutable item vector element accessor.
     */
    TRACCC_HOST_DEVICE
    item_t at(const link_type& link) { return items[link.first][link.second]; }

    /**
     * @brief Bounds-checking immutable item vector element accessor.
     */
    TRACCC_HOST_DEVICE
    const item_t at(const link_type& link) const {
        return items[link.first][link.second];
    }

    /**
     * @brief Mutable element accessor.
     */
    TRACCC_HOST_DEVICE
    element_view operator[](size_type i) { return {headers[i], items[i]}; }

    /**
     * @brief Immutable element accessor.
     */
    TRACCC_HOST_DEVICE
    const_element_view operator[](size_type i) const {
        return {headers[i], items[i]};
    }

    /**
     * @brief Return the size of the container.
     *
     * In principle, the size of the two internal vectors should always be
     * equal, but we can assert this at runtime for debug builds.
     */
    TRACCC_HOST_DEVICE
    size_type size(void) const { return headers.size(); }

    /**
     * @brief Reserve space in both vectors.
     */
    TRACCC_HOST
    void reserve(size_type s) {
        headers.reserve(s);
        items.reserve(s);
    }

    /**
     * @brief Resize space in both vectors.
     */
    TRACCC_HOST
    void resize(size_type s) {
        headers.resize(s);
        items.resize(s);
    }

    /**
     * @brief Push a header and a vector into the container.
     */
    template <typename h_prime, typename v_prime>
    TRACCC_HOST void push_back(h_prime&& new_header, v_prime&& new_items) {
        headers.push_back(std::forward<header_t>(new_header));
        items.push_back(
            std::forward<typename item_vector::value_type>(new_items));
    }

    /**
     * @brief Accessor method for the internal header vector.
     */
    TRACCC_HOST_DEVICE
    const header_vector& get_headers() const { return headers; }

    /**
     * @brief Non-const accessor method for the internal header vector.
     *
     * @warning Do not use this function! It is dangerous, and risks breaking
     * invariants!
     */
    TRACCC_HOST_DEVICE
    header_vector& get_headers() { return headers; }

    /**
     * @brief Accessor method for the internal item vector-of-vectors.
     */
    TRACCC_HOST_DEVICE
    const item_vector& get_items() const { return items; }

    /**
     * @brief Non-const accessor method for the internal item vector-of-vectors.
     *
     * @warning Do not use this function! It is dangerous, and risks breaking
     * invariants!
     */
    TRACCC_HOST_DEVICE
    item_vector& get_items() { return items; }

    /**
     * @breif Get number of items of jagged vector
     */
    uint64_t total_size() const {
        uint64_t ret = 0;
        for (auto& item : items) {
            ret += item.size();
        }
        return ret;
    }

    private:
    /// Headers information related to the objects in the event
    header_vector headers;

    /// All objects in the event
    item_vector items;
};

/// Convenience declaration for the container type to use in host code
template <typename header_t, typename item_t>
using host_container =
    container<header_t, item_t, vecmem::vector, vecmem::jagged_vector>;

/// Convenience declaration for the container type to use in device code
template <typename header_t, typename item_t>
using device_container = container<header_t, item_t, vecmem::device_vector,
                                   vecmem::jagged_device_vector>;

/// @name Types used to send data back and forth between host and device code
/// @{

/// Structure holding (some of the) data about the container in host code
template <typename header_t, typename item_t>
struct container_data {
    vecmem::data::vector_view<header_t> headers;
    vecmem::data::jagged_vector_data<item_t> items;
};

/// Structure holding (all of the) data about the container in host code
template <typename header_t, typename item_t>
struct container_buffer {
    vecmem::data::vector_buffer<header_t> headers;
    vecmem::data::jagged_vector_buffer<item_t> items;
};

/// Structure used to send the data about the container to device code
///
/// This is the type that can be passed to device code as-is. But since in
/// host code one needs to manage the data describing a
/// @c traccc::container either using @c traccc::container_data or
/// @c traccc::container_buffer, it needs to have constructors from
/// both of those types.
///
/// In fact it needs to be created from one of those types, as such an
/// object can only function if an instance of one of those types exists
/// alongside it as well.
///
template <typename header_t, typename item_t>
struct container_view {

    /// Constructor from a @c container_data object
    container_view(const container_data<header_t, item_t>& data)
        : headers(data.headers), items(data.items) {}

    /// Constructor from a @c container_buffer object
    container_view(const container_buffer<header_t, item_t>& buffer)
        : headers(buffer.headers), items(buffer.items) {}

    /// View of the data describing the headers
    vecmem::data::vector_view<header_t> headers;

    /// View of the data describing the items
    vecmem::data::jagged_vector_view<item_t> items;
};

/// Helper function for making a "simple" object out of the container
template <typename header_t, typename item_t>
inline container_data<header_t, item_t> get_data(
    host_container<header_t, item_t>& cc,
    vecmem::memory_resource* resource = nullptr) {
    return {{vecmem::get_data(cc.get_headers())},
            {vecmem::get_data(cc.get_items(), resource)}};
}

}  // namespace traccc
