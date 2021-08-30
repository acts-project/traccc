
#include "algebra/definitions/vecmem_array.hpp"

namespace traccc {

    using scalar = algebra::scalar;

    template <typename value_type, unsigned int kDIM>
    using darray = algebra::array_s<value_type, kDIM>;

    template <typename value_type>
    using dvector = algebra::vector_s<value_type>;

    template <typename key_type, typename value_type>
    using dmap = algebra::map_s<key_type, value_type>;

    template< class... types>
    using dtuple = algebra::tuple_s<types ...>;

    using algebra::operator*;
    using algebra::operator+;
    using algebra::operator-;

    namespace getter = algebra::getter;
    namespace vector = algebra::vector;

} //namespace traccc
