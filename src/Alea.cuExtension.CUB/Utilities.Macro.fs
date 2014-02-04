[<AutoOpen>]
module Alea.cuExtension.CUB.Utilities.Macro

let CUB_MAX a b = if a > b then a else b
let CUB_MIN a b = if a < b then a else b
let CUB_QUOTIENT_FLOOR x y = x / y
let CUB_QUOTIENT_CEILING x y = (x + y - 1) / y
let CUB_ROUND_UP_NEAREST x y = ((x + y - 1) / y) * y
let CUB_ROUND_DOWN_NEAREST x y = (x / y) * y