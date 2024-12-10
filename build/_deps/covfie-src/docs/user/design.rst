Library design
==============

The world in which we live is arguably three-dimensional. If you count time,
it's four-dimensional. If you look at the work machine leaning scientists do,
they work with data that exists in dozens of dimensions, if not thousands. This
presents a fundamental problem for programmers, who build software on computers
that are equipped with one-dimensional memory: as far as we have come since the
days of `Alan Turing <https://en.wikipedia.org/wiki/Turing_machine>`_, the
memories in our computers are still just tapes. The problem thus arises how we
map our higher-dimensional data onto our single-dimensional data, and --
perhaps even more importantly -- how we do that efficiently.

Principles
----------

In order to be as efficient and maintainable as possible, covfie is based upon
a set of core principes which guide its design. Understanding these principles
is important for understanding the rest of the library.

Compile-time knowledge

  Modern day compilers are incredible tools: they can issue powerful warnings
  and prevent us from shooting ourselves in the foot in a variety of different
  ways. However, the compiler cannot protect us as run-time. In covfie, we try
  to determine as much as possible about the use case of the vector field at
  compile time. covfie also allows absolutely no run-time polymorphism.

Compositionality

  In the `immortal words of Brian Beckman
  <https://www.youtube.com/watch?v=ZhuHCtR3xq8>`_, composition is *the* way to
  control complexity in software. Wherever we see complex software, it is our
  duty as programmers to break it down into smaller, composable pieces, and to
  find useful abstractions. covfie takes a compositional approach to vector
  fields which allows us to be as flexible as possible with as little code as
  possible.

Extensibility
  In the real world, library users almost always have wishes which library
  designers did not think of. In some libraries, this leads to insurmountable
  obstacles. We want covfie to be helpful in these cases: the library is
  designed with extensibility in mind, such that end users can easily use their
  own layouts, storage types, and interpolation algorithms if they so please.

Generality
  covfie was originally designed for high-energy physics to model magnetic
  fields in particle accelerator experiments, but we do not want covfie to be
  limited to just this purpose. Our library is designed to be generally
  applicable, so you can use if regardless of whether you're working with
  magnetic fields or ocean currents.

.. warning::
    What follows is a somewhat needlessly abstract and mathematical description
    of the problems with representing vector fields and how we solve those
    problems. Reading this section is not necessary to understand and use the
    library.

Vector fields
-------------

Finding a rigorous definition of what a vector field is is surprisingly
difficult. In the most general sense, we might take two vector spaces :math:`V`
and :math:`W` and define a vector field as a mapping:

.. math::
    V \to W

One interesting question is whether vector fields are necessarily endomorphic,
such that the vector space on the left side of the arrow is the same as the
vector space on the right. While vector fields of this structure are
well-behaved we do not assume that this is generally the case. Thus, a vector
field may map from one vector space to another.

Vector fields in a general sense are also a little too complex to model: some
rather unusual structures can be described a vector fields, and as such we will
restrict ourselves to so-called *coordinate spaces*, where the vector spaces on
either side of the arrow are tuples over some algebraic field. Thus, if we have
two algebraic fields :math:`F` and :math:`G` as well as two whole numbers
:math:`n` and :math:`m`, we can describe our vector field as follows:

.. math::
    F^n \to G^m

Finally, we will make one additional modification to this model: the structures
over which we define our coordinates need not be fields, they are only required
to be commutative semirings, such that we can model the natural numbers
:math:`\mathbb{N}_0` and the Booleans :math:`\mathbb{B}`, which are crucial for
modeling computer memory as we will see later.

Composition
-----------

.. _composition:

As we have discussed previously, our slightly expanded definition of what a
vector field is is given by two commutative semirings :math:`R` and :math:`S`
as well as two numbers :math:`n` and :math:`m` such that our vector field is
given as follows:

.. math::
    R^n \to S^m

Now, if we consider what our computer memory (we'll write :math:`M`) looks
like, it allows us to request a set of bits (of arbitrary length) at a certain
location in its one-dimensional tape-like memory. Thus, we could describe it as
a mapping of the following form:

.. math::
    M : \mathbb{N} \to \mathbb{B}^p

And therein lies the problem of designing a library for vector fields: the
mapping that we want to represent and the mapping that we have in our memory
are so different that it is not immediately obvious how to turn one into the
other. However, it turns out that the compositional nature of our library will
give us a way to model this in an elegant way.

The key insight here is that the functional arrow :math:`\to` that forms the
basis of our definition of a vector field is a profunctor. In other words, it
is a bifunctor that is covariant in its second argument (thus, :math:`A \to` is
a covariant functor) and contravariant in its first argument (meaning
:math:`\to B` is a contravariant functor). Therefore, we can apply additional
mapping functions to both the return value of the field (in a covariant
fashion) as well as the argument value (in a contravariant fashion). We will
define our vector fields in terms of *layers*, each of which contravariantly
mutates the input of the mapping and covariantly mutates the output. These
layers, described by the quaternary type constructor :math:`L` and a functor
:math:`F`, are pairs of two functions:

.. math::
    L : \mathrm{Functor}~F \Rightarrow (A' \to F(A), F(B) \to B')

In code, this can be described using general algebraic data types, such as in
this Haskell snippet:

.. code-block:: haskell

    data Layer i j k l where
        L :: Functor f
          => (a' -> f a)
          -> (f b -> b')
          -> L a' (f a) (f b) b'

It may be worth noting that both the contravariant and covariant parts of a
layer can be deconstructed further using :math:`F`-(co)algebras. Indeed, the
contravariant part can be constructed from a function :math:`f` of type
:math:`A' \to A`, and either an :math:`F`-coalgebra carried by :math:`A` or an
:math:`F`-coalgebra carried by :math:`A'`. In such a construction, :math:`f`
acts as a homomorphism between the two :math:`F`-coalgebras, and all the usual
rules for such a homomorphism apply. The covariant part of a layer can be
constructed using the exact dual of this idea, employing :math:`F`-algebras
instead of coalgebras.

To understand how we can use such a layer to transform an existing mapping
(such as our memory :math:`M`), we follow the principle described earlier of
applying the first function contravariantly and applying the second one
covariantly, such that applying a layer to an existing function has the
following type:

.. math::
    $_L : (A' \to F(A), F(B) \to B') \to (A \to B) \to A' \to B'

And the following corresponding implementation:

.. math::
    \begin{align}
    (f, g)~$_L~h = g \circ F(h) \circ f
    \end{align}

As a practical example, let's consider the case where we have our computer's
memory, :math:`M : \mathbb{N} \to \mathbb{B}^p`, and we have a layer that can
convert this to a vector field from three-dimensional reals to
three-dimensional reals, which would have the following type (the identity
functor :math:`\mathrm{Id}` is not relevant for the time being):

.. math::
    l : (\mathbb{R}^3 \to \mathrm{Id}(\mathbb{N}), \mathrm{Id}(\mathbb{B}^p) \to \mathbb{R}^3)

To layer this over :math:`M` -- giving :math:`l~$_L~M` -- we would receive an
object of type :math:`\mathbb{R}^3` as our input. This would be passed into the
contravariant part of our layer (which has type :math:`\mathbb{R}^3 \to
\mathrm{Id}(\mathbb{N})`) to produce an object of type
:math:`\mathrm{Id}(\mathbb{N})`. We then lift the mapping :math:`M` into the
identity functor to create :math:`\mathrm{Id}(M)`, converting the old mapping
of type :math:`\mathbb{N} \to \mathbb{B}^p` into a new mapping of type
:math:`\mathrm{Id}(\mathbb{N}) \to \mathrm{Id}(\mathbb{B}^p)`. Passing the
previously produced value of type :math:`\mathrm{Id}(\mathbb{N})` into this
mapping gives an object of type :math:`\mathrm{Id}(\mathbb{B}^p)`. Finally, we
can apply the covariant part of our layer, which has type
:math:`\mathrm{Id}(\mathbb{B}^p) \to \mathbb{R}^3` to that value to produce the
desired vector type :math:`\mathbb{R}^3`!

In code, the application of layers to functions is given as follows:

.. code-block:: haskell

    (|$|) ::
        Functor f
        => Layer a (f b) (f c) d
        -> (b -> c)
        -> (a -> d)
    (|$|) (Layer f g) h = g . (fmap h) . f

The magic of such layers is that they can be constructed through the
composition of simpler layers. Composition of layers is an operation with the
following type:

.. math::
    :nowrap:

    \begin{align}
    \circ_L :~&(A'' \to F(A'), F(B') \to B'')\\
    \to~&(A' \to G(A), G(B) \to B')\\
    \to~&(A'' \to (F \circ G)(A), (F \circ G)(B) \to B'')
    \end{align}

Such that:

.. math::
    (f_1, g_1) \circ_L (f_2, g_2) = (F(f_2) \circ f_1, g_1 \circ F(g_2))

This operation is conveniently broken down into a contravariant part and a
covariant part. The contravariant part mutates the input, on which the "outer"
layer, on the left of the composition operator, operates first. Thus, we first
apply the contravariant part of the outer layer (with type :math:`A'' \to
F(A')`) to the input, which creates a value in the :math:`F` functor, and we
therefore need to lift the contravariant side of the *inner* layer (with type
:math:`A' \to G(A)`) into the :math:`F` functor to be able to apply it to the
value created by the outer layer. The covariant part of the composition is
created in a very similar vein, but in reverse. That process is left as an
exercise to the reader. The equivalent code looks like this:

.. code-block:: haskell

    (|.|) :: (Functor f, Functor g)
        => Layer a1 (f a2) (f b2) b1
        -> Layer a2 (g a3) (g b3) b2
        -> Layer a1 ((Compose f g) a3) ((Compose f g) b3) b1
    (|.|) (Layer f1 g1) (Layer f2 g2) = Layer contra co
        where
            contra = (Compose . (fmap f2) . f1)
            co     = (g1 . (fmap g2) . getCompose)

The identity element of these layers under composition is given as a layer
where the contravariant component lifts a value into the identity functor, and
a covariant component which extracts the value from it. Thus, we might define
the identity layer as follows:

.. math::
    \mathrm{id}_L = (\lambda x . \mathrm{Id}(x), \lambda \mathrm{Id}(x) . x)

Or, once again, in code:

.. code-block:: haskell

    idLayer = (Identity, runIdentity)

Abstract example
----------------

Let's consider the case where we want to model a magnetic field, which is a
mapping from three-dimensional real coordinates to three-dimensional vectors.
Thus, the type of such a magnetic field is a mapping :math:`\mathbb{R}^3 \to
\mathbb{R}^3`. From the previous section we know that we can achieve this by
finding a layer :math:`l` of the following type:

.. math::
    l : \mathrm{Functor}~F\Rightarrow (\mathbb{R}^3 \to F(\mathbb{N}), F(\mathbb{B}^p) \to \mathbb{R}^3)

The key insight is that constructing the pair of functions :math:`l` is best
achieved compositionally; if we wrote these functions in one go, we would be
unable to re-use them for anything else. Thus, we will consider how to
construct them compositionally, from a set of simpler layers.

The first layer we will consider is the data type. Remember that the basic
definition of our memory is a mapping from an address to a series of bits. The
first layer of our composition shall thus be a layer that interprets those bits
as the real numbers we desire. The type of this layer is as follows:

.. math::
    l_1 : (\mathbb{N} \to \mathrm{Id}(\mathbb{N}), \mathrm{Id}(\mathbb{B^p}) \to \mathbb{R}^3)

Assuming we want to model our real numbers as 32-bit IEEE 754 floating point
numbers, we will need to consider that each output vector, consisting of three
of such numbers, will consist of a total of 96 bits. Thus, to move forward one
vector in memory, we need to skip over 96 bits. In other words, to access the
:math:`n`-th vector, we need to look up the memory starting at the
:math:`(96n)`-th bit. Then, we need to interpret those 96 bits as three
integers. In lieu of convenient mathematical notation, we will define our first
layer as follows:

.. math::
    l_1 = (\lambda x.\mathrm{Id}(96x), \lambda \mathrm{Id}(x).\mathtt{reinterpret\_cast<float[3]>}(x))

This gives us a mapping from one-dimensional coordinates to three-dimensional
vectors: a step in the right direction, but not what we want quite yet. We will
need to compose some kind of layer that can take three-dimensional coordinates
and interpret them as one-dimensional coordinates. In this layer, we don't need
to modify the output at all. Thus, we arrive at the following type:

.. math::
    l_2 : (\mathbb{N}^3 \to \mathrm{Id}(\mathbb{N}), \mathrm{Id}(\mathbb{R}^3) \to \mathbb{R}^3)

We'll assume that the size of our field in each
direction is known as :math:`N`. Adhering to a `column-major storage order
<https://en.wikipedia.org/wiki/Row-_and_column-major_order>`_, we can define
our next layer as follows:

.. math::
    l_2 = (\lambda (c_1, c_2, c_3).\mathrm{Id}\left(\sum_{k=1}^3\left(\prod_{l = k + 1}^3 N_l\right)c_k\right), \lambda\mathrm{Id}(x).x)

Next, we will want to prevent our vector field from going out of bounds. In
particular, we will want to control what happens in that case. Let's assume
that we want any accesses that go outsize our :math:`N_1 \times N_2 \times N_3`
mapping to retun the zero-vector instead of causing some sort of error. This is
sometimes known as *clamping*. Such a layer presents an interesting challenge
because it is the first layer that does not employ the identity functor.
Rather, the type of this layer is the following:

.. math::
    l_3 : \left(\mathbb{N}^3 \to \mathrm{Maybe}(\mathbb{N} ^3), \mathrm{Maybe}(\mathbb{R}^3) \to \mathbb{R}^3\right)

Intuitively, if the requested coordinate lies inside of our mapping, the
contravariant side of this layer will produce an extant value, and the
underlying layer will be called to operate on it as normal. However, if the
coordinate lies outside of the requested mapping, a non-extant value is
produced, which is threaded through any underlying layers (effectively
performing no computation) and the non-extant value will be returned to the
covariant side of the layer, which can then return the default value! The layer
is given as follows:

.. math::
    :nowrap:

    \begin{align}
    l_3 = \bigg(&\lambda (c_1, c_2, c_3). \begin{cases}
        \mathrm{Just}((c_1, c_2, c_3)) & \mathrm{if}~c_1 < N_1 \wedge c_2 < N_2 \wedge c_3 < N_3\\
        \mathrm{Nothing} & \mathrm{otherwise}
        \end{cases},\\
        &\lambda x . \begin{cases}
        (r_1, r_2, r_3) & \mathrm{if}~x = \mathrm{Just}((r_1, r_2, r_3))\\
        (0, 0, 0) & \mathrm{otherwise}
        \end{cases}\bigg)
    \end{align}

Next, let's get rid of the constraint that we can only access our vector field
at integer coordinates. To do so, we will need to construct some sort of
interpolation method. Using a nearest-neighbour method this is rather simple.
The type of such a layer would be:

.. math::
    l'_4 : (\mathbb{R}^3 \to \mathrm{Id}(\mathbb{N}^3), \mathrm{Id}(\mathbb{R}^3) \to \mathbb{R}^3)

And it's implementation might look something like this, where we simply round
the coordinates on the contravariant side and leave the covariant side
unchanged:

.. math::
    l'_4 = (\lambda (c_1, c_2, c_3).\mathrm{Id}((\lfloor c_1 \rceil, \lfloor c_2 \rceil, \lfloor c_3 \rceil)), \lambda \mathrm{Id}(x) . x)

But this is boring. A more interesting example that showcases the true power of
our approach is linear interpolation. This requires us to access the underlying
function eight times for every access, which we can model elegantly using our
functor-based approach. Our new interpolation method has the following layer
type, noting that the creation of 8-tuples is our functor:

.. math::
    l_4 : (\mathbb{R}^3 \to (\mathbb{N}^3)^8, (\mathbb{R}^3)^8 \to \mathbb{R}^3)

In this case, we will omit the full equation for the layer, but intuitively the
contravariant side takes one real-valued coordinate and produces the eight
integer-valid coordinates which are closest to it. On the covariant side, we
find eight resulting vectors of which we take the weighted average to find our
final vector.

We'll now add one final layer. As it stands, the coordinates we feed our vector
field have no geometric meaning. They are -- for the lack of a better term --
coordinates in our computer's memory, rather than in space. Thankfully, this
problem is easily solved: we will add a layer that transforms geometrically
meaningful coordinates into the ones our current field accepts through an
affine transformation. The type of this layer is simple:

.. math::
    l_5 : (\mathbb{R}^3 \to \mathrm{Id}(\mathbb{R}^3), \mathrm{Id}(\mathbb{R}^3) \to \mathbb{R}^3)

The definition is not much more complex, if we assume that we are in possession
of some affine transformation matrix :math:`A`:

.. math::
    l_5 = (\lambda x . \mathrm{Id}(Ax), \lambda \mathrm{Id}(x).x)

Finally, we are ready to construct our magnetic field :math:`B`. First, let's
compose all of the layers into one layer :math:`l_B`. This is achieved very
simply through the operators we defined earlier:

.. math::
    l_B = l_5 \circ_L l_4 \circ_L l_3 \circ_L l_2 \circ_L l_1

Which we can then apply to the machine's memory to obtain our magnetic field:

.. math::
    B = l_B~$_L~M

Note that this notion of composition is associative, such that the following
are all equivalent:

.. math::
    :nowrap:

    \begin{align}
    B &= (l_5 \circ_L l_4 \circ_L l_3 \circ_L l_2 \circ_L l_1)~$_L~M\\
    B &= (l_5 \circ_L l_4) \circ_L (l_3 \circ_L l_2 \circ_L l_1)~$_L~M\\
    B &= l_5 \circ_L (l_4 \circ_L l_3 \circ_L l_2) \circ_L l_1~$_L~M\\
    B &= (((l_5 \circ_L l_4) \circ_L l_3) \circ_L l_2) \circ_L l_1~$_L~M\\
    B &= l_5 \circ_L (l_4 \circ_L (l_3 \circ_L (l_2 \circ_L l_1)))~$_L~M
    \end{align}

And so forth. In addition, composing one layer with another and then applying
that layer to a mapping is equivalent to applying the innermost layer to the
mapping and then applying the outermost layer, such that:

.. math::
    (l_3 \circ_L l_2 \circ_L l_1)~$_L~M = (l_3 \circ_L l_2)~$_L~(l_1 $_L M) = l_3~$_L~(l_2~$_L~(l_1~$_L~M))

The final type of our magnetic field is as follows:

.. math::
    :nowrap:

    \begin{align}
    l_B : (&\mathbb{R}^3 \to (\mathrm{Id}\circ -^8 \circ \mathrm{Id} \circ \mathrm{Maybe} \circ \mathrm{Id} \circ \mathrm{Id})(\mathbb{N}),\\
        &(\mathrm{Id}\circ -^8 \circ \mathrm{Id} \circ \mathrm{Maybe} \circ \mathrm{Id} \circ \mathrm{Id})(\mathbb{B}^p) \to \mathbb{R}^3)
    \end{align}

And while that seems like quite a mouth full, it is worth noting that the
composition of functors :math:`\mathrm{Id}\circ -^8 \circ \mathrm{Id} \circ
\mathrm{Maybe} \circ \mathrm{Id} \circ \mathrm{Id}` is itself a well-behaved
functor, meaning we can plug it into our definition of :math:`$_L` without
worrying about it: composition has handled this complexity for us.
