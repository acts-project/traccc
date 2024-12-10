{-
 - SPDX-PackageName: "covfie, a part of the ACTS project"
 - SPDX-FileCopyrightText: 2022 CERN
 -
 - SPDX-License-Identifier: MPL-2.0
 -}

{-# LANGUAGE GADTs #-}

import Data.Functor.Compose
import Data.Functor.Identity

{-
 - Our layers are declared as quaternary type constructors, where the type
 - "Layer i j k l" represents a layer that contravariantly converts type i to
 - j, performs the inner function (taking a j and returning a k), and then
 - covariantly converts the k to an l. Layers are defined as GADTs because this
 - allows us to enforce that j and k live inside the same functor.
 -}
data Layer i j k l where
    Layer :: Functor f
        => (a' -> f a)
        -> (f b -> b')
        -> Layer a' (f a) (f b) b'

{-
 - This function models the application of a layer to a normal function. Doing
 - so creates a new function, where the input is first processed by the
 - contravariant part of the layer, the inner function is then called, and the
 - result of that computation is passed to the covariant part of the layer.
 -}
(|$|) ::
    Functor f
    => Layer a (f b) (f c) d
    -> (b -> c)
    -> (a -> d)
(|$|) (Layer f g) h = g . (fmap h) . f

(|-|) :: (Functor f, Functor g)
    => Layer a1 (f a2) (f b2) b1
    -> Layer a2 (g a3) (g b3) b2
    -> Layer a1 ((Compose f g) a3) ((Compose f g) b3) b1
(|-|) (Layer f1 g1) (Layer f2 g2) = Layer contra co
    where
        contra = (Compose . (fmap f2) . f1)
        co     = (g1 . (fmap g2) . getCompose)

{-
 - The remainder of this program briefly shows how layers can be composed.
 -
 - This first layer, layer 1, accepts a string contravariantly and reads an
 - integer from it. This integer is passed to whatever function it is applied
 - to, which returns an integer, which is then converted to a floating point
 - number.
 -}
layer1 :: Layer String (Identity Integer) (Identity Integer) Float
layer1 = Layer (Identity . read) (fromInteger . runIdentity)

{-
 - The second layer accepts a string contravariantly and appends it to itself,
 - and then applies the inner function to it. The inner function is assumed to
 - return a floating point number in the identity functor, to which 5.0 is
 - added.
 -}
layer2 :: Layer String (Identity String) (Identity Float) Float
layer2 = Layer (Identity . (\x -> x ++ x)) ((+5.0) . runIdentity)

{-
 - The third layer accepts an integer, and produces three strings: the first is
 - a string representation of the integer itself, the second is a string
 - representing the integer plus one, and the third is a string representing
 - the integer plus two. Each of these strings is passed to the inner
 - computation, which produces a float. Thus, we get a list of floats as our
 - covariant input, which we sum up.
 -}
layer3 :: Layer Integer [String] [Float] Float
layer3 = Layer (\x -> [show x, show (x + 1), show (x + 2)]) (sum)

{-
 - In the main function, we compose the three layers above, creating a new
 - layer which we apply to a function which doubles its input. In effect, this
 - creates the following chain of computation when applied to the number 5:
 -
 - :: 5 (start)
 - -> ["5", "6", "7"] (layer 1)
 - -> ["55", "66", "77"] (layer 2)
 - -> [55, 66, 77] (layer 3)
 - -> [110, 132, 154] (core)
 - -> [110.0, 132.0, 154.0] (layer 3)
 - -> [115.0, 137.0, 159.0] (layer 2)
 - -> 411.0 (layer 1)
 -}
main :: IO ()
main = putStrLn . show . (layer3 |-| layer2 |-| layer1 |$| (*2)) $ 5
