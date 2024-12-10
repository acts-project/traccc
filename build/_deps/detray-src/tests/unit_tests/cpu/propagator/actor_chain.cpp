/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "detray/propagator/actor_chain.hpp"

#include "detray/definitions/units.hpp"
#include "detray/propagator/base_actor.hpp"

// GTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <sstream>
#include <string>

using namespace detray;

/// Actor that prints its call chain and subject data
struct print_actor : detray::actor {

    /// State keeps an internal string representation
    struct state {
        std::stringstream stream{};

        std::string to_string() const { return stream.str(); }
    };

    /// Actor implementation: append call notification to internal string
    template <typename propagator_state_t>
    void operator()(state &printer_state,
                    const propagator_state_t & /*p_state*/) const {
        printer_state.stream << "[print actor]:";
    }

    /// Observing actor implementation: append call notification to internal
    /// string
    template <typename subj_state_t, typename propagator_state_t>
    void operator()(state &printer_state, const subj_state_t &subject_state,
                    const propagator_state_t & /*p_state*/) const {
        printer_state.stream << "[print actor obs "
                             << subject_state.buffer.back() << "]:";
    }
};

/// Example actor that couts the number of elements in its buffer
template <template <typename...> class vector_t>
struct example_actor : detray::actor {

    /// actor state
    struct state {

        // Keep dynamic data per propagation stream
        vector_t<float> buffer = {};
    };

    /// Actor implementation: Counts vector elements
    template <typename propagator_state_t>
    void operator()(state &example_state,
                    const propagator_state_t & /*p_state*/) const {
        example_state.buffer.push_back(
            static_cast<float>(example_state.buffer.size()));
    }

    /// Observing actor implementation: Counts vector elements (division)
    template <typename propagator_state_t>
    void operator()(state &example_state, const state &subject_state,
                    const propagator_state_t & /*p_state*/) const {
        example_state.buffer.push_back(
            static_cast<float>(subject_state.buffer.size()) * 0.1f);
    }

    /// Observing actor implementation to printer: do nothing
    template <typename subj_state_t, typename propagator_state_t>
    requires(!std::is_same_v<subj_state_t, state>) void operator()(
        state & /*example_state*/, const subj_state_t & /*subject_state*/,
        const propagator_state_t & /*p_state*/) const { /*Do nothing*/
    }
};

using example_actor_t = example_actor<std::vector>;
// Implements example_actor with two print observers
using composite1 =
    composite_actor<dtuple, example_actor_t, print_actor, print_actor>;
// Implements example_actor with one print observer
using composite2 = composite_actor<dtuple, example_actor_t, print_actor>;
// Implements example_actor through composite2 and has composite1 as observer
using composite3 = composite_actor<dtuple, example_actor_t, composite1>;
// Implements example_actor through composite2<-composite3 with composite1 obs.
using composite4 = composite_actor<dtuple, example_actor_t, composite1>;

/* Test chaining of multiple actors
 * The chain goes as follows (depth first):
 *                          example_actor1
 *                              1.|
 *                          observer_lvl1 (print)
 *                              2.|
 *                          observer_lvl2 (example_actor observing print actor)
 *                      3./     5.|     6.\
 *            observer_lvl3 example_actor2 print
 *          (example_actor3)
 *               4.|
 *               print
 */
using observer_lvl3 = composite_actor<dtuple, example_actor_t, print_actor>;
using observer_lvl2 = composite_actor<dtuple, example_actor_t, observer_lvl3,
                                      example_actor_t, print_actor>;
using observer_lvl1 = composite_actor<dtuple, print_actor, observer_lvl2>;
using chain = composite_actor<dtuple, example_actor_t, observer_lvl1>;

// Test the actor chain on some dummy actor types
GTEST_TEST(detray_propagator, actor_chain) {

    // The actor states (can be reused between actors)
    example_actor_t::state example_state{};
    print_actor::state printer_state{};

    // Aggregate actor states to be able to pass them through the chain
    auto actor_states = detray::tie(example_state, printer_state);

    // Propagator state
    struct empty_prop_state {};
    empty_prop_state prop_state{};

    // Chain of actors
    using actor_chain_t = actor_chain<dtuple, example_actor_t, composite1,
                                      composite2, composite3, composite4>;
    // Run
    actor_chain_t run_actors{};
    run_actors(actor_states, prop_state);

    ASSERT_TRUE(printer_state.to_string().compare(
                    "[print actor obs 1]:[print actor obs 1]:[print actor obs "
                    "2]:[print actor obs 0.4]:[print actor obs 0.4]:[print "
                    "actor obs 0.6]:[print actor obs 0.6]:") == 0)
        << "Printer call chain: " << printer_state.to_string() << std::endl;

    // Test chaining of multiple actors

    // Reset example actor state
    example_state.buffer.clear();
    printer_state.stream.str("");
    printer_state.stream.clear();

    // Run the chain
    actor_chain<dtuple, chain> run_chain{};
    run_chain(actor_states, prop_state);

    ASSERT_TRUE(printer_state.to_string().compare(
                    "[print actor obs 0]:[print actor obs 0.1]:[print actor "
                    "obs 0.2]:") == 0)
        << "Printer call chain: " << printer_state.to_string() << std::endl;
}
