/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/details/reverse_iterator.hpp"
#include "vecmem/containers/details/static_vector_traits.hpp"
#include "vecmem/utils/type_traits.hpp"
#include "vecmem/utils/types.hpp"

// System include(s).
#include <cstddef>
#include <type_traits>

namespace vecmem {

/// Class mimicking @c std::vector on top of a fixed sized array
///
/// This can come in handy when needing vector arithmetics in device code,
/// without resorting to heap allocations.
///
/// The type does come with a significant limitation over @c std::vector.
/// It has a maximal/fixed size that needs to be chosen at compile time.
///
template <typename TYPE, std::size_t MAX_SIZE>
class static_vector {

public:
    /// @name Type definitions, mimicking @c std::vector
    /// @{

    /// Type of the array elements
    typedef TYPE value_type;
    /// Size type for the array
    typedef std::size_t size_type;
    /// Pointer difference type
    typedef std::ptrdiff_t difference_type;

    /// The maximal size of the vector
    static constexpr size_type array_max_size = MAX_SIZE;
    /// The size of the vector elements
    static constexpr size_type value_size = sizeof(value_type);
    /// Type of the array holding the payload of the vector elements
    typedef typename details::static_vector_type<
        char, array_max_size * value_size>::type array_type;

    /// Value reference type
    typedef value_type& reference;
    /// Constant value reference type
    typedef const value_type& const_reference;
    /// Value pointer type
    typedef value_type* pointer;
    /// Constant value pointer type
    typedef const value_type* const_pointer;

    /// Forward iterator type
    typedef pointer iterator;
    /// Constant forward iterator type
    typedef const_pointer const_iterator;
    /// Reverse iterator type
    typedef vecmem::details::reverse_iterator<iterator> reverse_iterator;
    /// Constant reverse iterator type
    typedef vecmem::details::reverse_iterator<const_iterator>
        const_reverse_iterator;

    /// @}

    /// @name Constructors and destructor, mimicking @c std::vector
    /// @{

    /// Default constructor
    VECMEM_HOST_AND_DEVICE
    static_vector();
    /// Construct a vector with a specific size
    VECMEM_HOST_AND_DEVICE
    explicit static_vector(size_type size,
                           const_reference value = value_type());
    /// Construct a vector with values coming from a pair of iterators
    template <
        typename InputIt,
        std::enable_if_t<details::is_iterator_of<InputIt, value_type>::value,
                         bool> = true>
    VECMEM_HOST_AND_DEVICE static_vector(InputIt other_begin, InputIt other_end)
        : m_size(0), m_elements() {

        assign<InputIt>(other_begin, other_end);
    }
    /// Copy constructor
    VECMEM_HOST_AND_DEVICE
    static_vector(const static_vector& parent);

    /// Destructor
    VECMEM_HOST_AND_DEVICE
    ~static_vector();

    /// @}

    /// @name Vector element access functions
    /// @{

    /// Return a specific element of the vector in a "safe way" (non-const)
    VECMEM_HOST_AND_DEVICE
    reference at(size_type pos);
    /// Return a specific element of the vector in a "safe way" (const)
    VECMEM_HOST_AND_DEVICE
    const_reference at(size_type pos) const;

    /// Return a specific element of the vector (non-const)
    VECMEM_HOST_AND_DEVICE
    reference operator[](size_type pos);
    /// Return a specific element of the vector (const)
    VECMEM_HOST_AND_DEVICE
    const_reference operator[](size_type pos) const;

    /// Return the first element of the vector (non-const)
    VECMEM_HOST_AND_DEVICE
    reference front();
    /// Return the first element of the vector (const)
    VECMEM_HOST_AND_DEVICE
    const_reference front() const;

    /// Return the last element of the vector (non-const)
    VECMEM_HOST_AND_DEVICE
    reference back();
    /// Return the last element of the vector (const)
    VECMEM_HOST_AND_DEVICE
    const_reference back() const;

    /// Access the underlying memory array (non-const)
    VECMEM_HOST_AND_DEVICE
    pointer data();
    /// Access the underlying memory array (const)
    VECMEM_HOST_AND_DEVICE
    const_pointer data() const;

    /// @}

    /// @name Payload modification functions
    /// @{

    /// Assign new values to the vector
    VECMEM_HOST_AND_DEVICE
    void assign(size_type count, const_reference value);
    /// Assign new values to the vector
    template <
        typename InputIt,
        std::enable_if_t<details::is_iterator_of<InputIt, value_type>::value,
                         bool> = true>
    VECMEM_HOST_AND_DEVICE void assign(InputIt other_begin, InputIt other_end) {

        // Remove all previous elements.
        clear();

        // Create copies of all of the elements one-by-one. It's very
        // inefficient, but we can't make any assumptions about the type of the
        /// input iterator eceived by this function.
        for (InputIt itr = other_begin; itr != other_end; ++itr) {
            construct(m_size++, *itr);
        }
    }

    /// Insert a new element into the vector
    VECMEM_HOST_AND_DEVICE
    iterator insert(const_iterator pos, const_reference value);
    /// Insert an element N times into the vector
    VECMEM_HOST_AND_DEVICE
    iterator insert(const_iterator pos, size_type count, const_reference value);
    /// Insert a list of elements into the vector
    template <
        typename InputIt,
        std::enable_if_t<details::is_iterator_of<InputIt, value_type>::value,
                         bool> = true>
    iterator insert(const_iterator pos, InputIt other_begin,
                    InputIt other_end) {

        // Find the index of this iterator inside of the vector.
        auto id = element_id(pos);

        // Insert the elements one by one. It's very inefficient, but we can't
        // make any assumptions about the type of the input iterator received
        // by this function.
        const_iterator self_itr = pos;
        for (InputIt other_itr = other_begin; other_itr != other_end;
             ++other_itr, ++self_itr) {
            insert(self_itr, *other_itr);
        }

        // Return an iterator to the first inserted element.
        return id.m_ptr;
    }

    /// Insert a new element into the vector
    template <typename... Args>
    VECMEM_HOST_AND_DEVICE iterator emplace(const_iterator pos, Args&&... args);
    /// Add a new element at the end of the vector
    template <typename... Args>
    VECMEM_HOST_AND_DEVICE reference emplace_back(Args&&... args);

    /// Add a new element at the end of the vector
    VECMEM_HOST_AND_DEVICE
    void push_back(const_reference value);

    /// Remove one element from the vector
    VECMEM_HOST_AND_DEVICE
    iterator erase(const_iterator pos);
    /// Remove a list of elements from the vector
    VECMEM_HOST_AND_DEVICE
    iterator erase(const_iterator first, const_iterator last);
    /// Remove the last element of the vector
    VECMEM_HOST_AND_DEVICE
    void pop_back();

    /// Clear the vector
    VECMEM_HOST_AND_DEVICE
    void clear();
    /// Resize the vector
    VECMEM_HOST_AND_DEVICE
    void resize(std::size_t new_size);
    /// Resize the vector and fill any new elements with the specified value
    VECMEM_HOST_AND_DEVICE
    void resize(std::size_t new_size, const_reference value);

    /// @}

    /// @name Iterator providing functions
    /// @{

    /// Return a forward iterator pointing at the beginning of the vector
    VECMEM_HOST_AND_DEVICE
    iterator begin();
    /// Return a constant forward iterator pointing at the beginning of the
    /// vector
    VECMEM_HOST_AND_DEVICE
    const_iterator begin() const;
    /// Return a constant forward iterator pointing at the beginning of the
    /// vector
    VECMEM_HOST_AND_DEVICE
    const_iterator cbegin() const;

    /// Return a forward iterator pointing at the end of the vector
    VECMEM_HOST_AND_DEVICE
    iterator end();
    /// Return a constant forward iterator pointing at the end of the vector
    VECMEM_HOST_AND_DEVICE
    const_iterator end() const;
    /// Return a constant forward iterator pointing at the end of the vector
    VECMEM_HOST_AND_DEVICE
    const_iterator cend() const;

    /// Return a reverse iterator pointing at the end of the vector
    VECMEM_HOST_AND_DEVICE
    reverse_iterator rbegin();
    /// Return a constant reverse iterator pointing at the end of the vector
    VECMEM_HOST_AND_DEVICE
    const_reverse_iterator rbegin() const;
    /// Return a constant reverse iterator pointing at the end of the vector
    VECMEM_HOST_AND_DEVICE
    const_reverse_iterator crbegin() const;

    /// Return a reverse iterator pointing at the beginning of the vector
    VECMEM_HOST_AND_DEVICE
    reverse_iterator rend();
    /// Return a constant reverse iterator pointing at the beginning of the
    /// vector
    VECMEM_HOST_AND_DEVICE
    const_reverse_iterator rend() const;
    /// Return a constant reverse iterator pointing at the beginning of the
    /// vector
    VECMEM_HOST_AND_DEVICE
    const_reverse_iterator crend() const;

    /// @}

    /// @name Capacity checking/modyfying functions
    /// @{

    /// Check whether the vector is empty
    VECMEM_HOST_AND_DEVICE
    bool empty() const;
    /// Return the number of elements in the vector
    VECMEM_HOST_AND_DEVICE
    size_type size() const;
    /// Return the maximum (fixed) number of elements in the vector
    VECMEM_HOST_AND_DEVICE
    size_type max_size() const;
    /// Return the current (fixed) capacity of the vector
    VECMEM_HOST_AND_DEVICE
    size_type capacity() const;
    /// Reserve additional storage for the vector
    VECMEM_HOST_AND_DEVICE
    void reserve(size_type new_cap);

    /// @}

private:
    /// Construct a new vector element
    VECMEM_HOST_AND_DEVICE
    void construct(size_type pos, const_reference value);
    /// Destruct a vector element
    VECMEM_HOST_AND_DEVICE
    void destruct(size_type pos);

    /// Helper type for identifying an element in the array
    struct ElementId {
        size_type m_index;
        pointer m_ptr;
    };  // struct ElementId
    /// Get the relevant identifiers of an element using an iterator
    VECMEM_HOST_AND_DEVICE
    ElementId element_id(const_iterator pos);

    /// Size of the vector
    size_type m_size;
    /// Array that holds the PoD elements of the vector
    array_type m_elements;

};  // class static_vector

}  // namespace vecmem

// Include the implementation.
#include "vecmem/containers/impl/static_vector.ipp"
