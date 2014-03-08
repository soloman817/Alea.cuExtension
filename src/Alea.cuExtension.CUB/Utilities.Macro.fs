[<AutoOpen>]
module Alea.cuExtension.CUB.Utilities.Macro

let [<ReflectedDefinition>] CUB_MAX a b = if a > b then a else b
let [<ReflectedDefinition>] CUB_MIN a b = if a < b then a else b
let [<ReflectedDefinition>] CUB_QUOTIENT_FLOOR x y = x / y
let [<ReflectedDefinition>] CUB_QUOTIENT_CEILING x y = (x + y - 1) / y
let [<ReflectedDefinition>] CUB_ROUND_UP_NEAREST x y = ((x + y - 1) / y) * y
let [<ReflectedDefinition>] CUB_ROUND_DOWN_NEAREST x y = (x / y) * y