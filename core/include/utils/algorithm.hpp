/**
 * TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc {
/**
 * @brief Unified algorithm semantics which convert an input to an output.
 *
 * This class provides a single, unified semantic for algorithms that can be
 * used throughout the traccc code. This virtual class is templated with an
 * input type and an output type, and any class implementing it is expected
 * to be able to transform the input type into the output type in one way or
 * another.
 *
 * @param I The input type of the algorithm
 * @param O The output type of the algorithm
 */
template <typename I, typename O>
class algorithm {
    public:
    using input_type = I;
    using output_type = O;

    /**
     * @brief Execute the algorithm, returning by value.
     *
     * Turn the object into a callable, accepting only an input and
     * returning the output through value return.
     *
     * @param[in] i The input value to the algorithm
     * @return An instance of the specified output type
     */
    virtual output_type operator()(const input_type& i) const = 0;

    /**
     * @brief Execute the algorithm, returning by input reference.
     *
     * Turn the object into a callable, accepting an input as well as an
     * output parameter to write to.
     *
     * @param[in] i The input value to the algorithm
     * @param[out] o The output of the algorithm
     */
    virtual void operator()(const input_type& i, output_type& o) const = 0;
};
}  // namespace traccc
