Vector field backends
=====================

In covfie, the majority of the behaviour of your vector field is determined by
the field :dfn:`backend`. In covfie, backends are divided into two categories:
initial backends and composite backends.

Initial backends
----------------

An :dfn:`initial backend` is a backend that is not composed of transformers.
Initial backends represent basic functionality that cannot -- for reasons of
practicality or performance -- be decomposed into smaller components. Initial
backends can be used directly, but often lack the necessary functionality to
model real-world vector fields. All initial backends have the kind :math:`*`.
Of course, many initial backends have additional type-level parameters which
influence their behaviour; these parameters need to be partially applied before
a usable initial backend is created.

Memory backends
~~~~~~~~~~~~~~~

Constant backends
~~~~~~~~~~~~~~~~~

Analytical backends
~~~~~~~~~~~~~~~~~~~

Transformers
------------

Backend transformers are useless on their own, but can be used to add
additional functionality to existing backends. Backend transformers form the
core of covfie's compositional nature. What follows is a selection of backend
transformers which shows off the power of these types. All transformers have
the kind :math:`* \to *`. As with initial backends, some backend transformers
have additional type-level parameters which need to be partially applied to
ensure the correct kind. After applying a backend transformer to an existing
(initial or composite) backend, a composite backend with kind :math:`*` is
created.

Please note that the following categories are only provided as guidelines;
there is little to no in-code enforcements of these categories, and it is
possible to create arbitrary backends which may fall outside of this taxonomy.

Storage order backends
~~~~~~~~~~~~~~~~~~~~~~

Storage order backends provide a translation layer between multi-dimensional
inputs and the single-dimensional inputs our memory needs. In general, storage
order backends have the type :math:`\forall a : (\mathbb{N}^n \to
\mathrm{Id}(\mathbb{N}), \mathrm{Id}(a) \to a)`. The covariant part of a
storage order backend is virtually always the embellished identity function,
and the contravariant part is a function which maps multidimensional indices
onto single dimensions.

Row-major and column-major layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Morton curve layout
^^^^^^^^^^^^^^^^^^^



Clamping backends
~~~~~~~~~~~~~~~~~

Clamping backends allow the user to control what happens when a memory access
goes out of bounds, possibly in multiple dimensions. Generally, clamping
backends have the type :math:`\forall a, b : (a \to \mathrm{Maybe}(a),
\mathrm{Maybe}(b) \to b)`, although some simpler clamping backends may take the
form :math:`\forall a, b : (a \to \mathrm{Id}(a), \mathrm{Id}(b) \to c)`. The
first type is used in cases where an out-of-bounds error is considered an
error, and must be compensated using some default value. The second form can be
used when an out-of-bounds input can be transformed to a new, valid input.

Interpolation backends
~~~~~~~~~~~~~~~~~~~~~~

Geometric backends
~~~~~~~~~~~~~~~~~~

Run-time information
--------------------

Practical composition
---------------------

In the :ref:`design <composition>` chapter of the user guide, we detail how
memory transformers can be freely composed with one another. While this
principle guides the design of our C++ code, the non-expressive nature of the
C++ type system requires us to make a few compromises.

The most obvious issue is that C++ does not allow infix operators to be defined
at the type level, which severely limits our ability to express composition.
The syntax which we can use in Haskell is not permissible in C++:

.. code-block:: haskell

    layer3 |-| layer2 |-| layer1 |$| (*2)

Rather, C++ permits us three ways to compose transformers, all of which are
non-ideal. The first, and probably most common method of composing transformers
is simply sequential application. Recall that the following are equivalent:

.. math::

    (l_1 \circ_L l_2 \circ_L l_3)~$_L~l_0 = (l_1 \circ_L l_2)~$_L~(l_3~$_L~l_0) = l_1~$_L~(l_2~$_L~(l_3~$_L~l_0))

In C++, we might compose layers through repeated application in the following
way:

.. code-block:: cpp

    using l0 = covfie::backend::constant<...>;
    using l1 = covfie::backend::my_transformer<l0>;
    using l2 = covfie::backend::my_interpolator<l1>;
    using l3 = covfie::backend::my_affine<l2>;

Alternatively, we can construct a new type constructor which applies multiple
transformer layers to the same initial backend:

.. code-block:: cpp

    template<typename T>
    using l123 = covfie::backend::my_affine<
        covfie::backend::my_interpolator<
            covfie::backend::my_transformer<
                T
            >
        >
    >;

    // Equivalent to l3 in the previous example
    using l3 = l123<covfie::backend::constant<...>>;

Finally, it is possible to compose a set of transformer layers variadically, as
follows:

.. code-block:: cpp

    template<
        template <typename> typename T,
        template <typename> typename ... Ts
    >
    struct compose {
        template<typename I>
        using type = std::conditional_t<
            (sizeof...(Ts) > 0),
            T<compose<Ts...>::type<I>>,
            T<I>
        >;
    };

    // Once again, equivalent to what is shown above
    using l3 = compose<
        covfie::backend::my_affine
        covfie::backend::my_interpolator
        covfie::backend::my_transformer
    >::type<covfie::backend::constant<...>>;

These three approaches are equivalent, and you are free to pick whichever fits
your project the best.
