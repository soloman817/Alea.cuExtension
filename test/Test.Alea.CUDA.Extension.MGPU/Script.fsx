#r "System"
open System
open System.Diagnostics
let stopwatch = Stopwatch()
let rng = System.Random()



let mutable x = 0;
let mutable coop = 2
while coop <= 256 do
    coop <- coop * 2
    x <- x + 1
printfn "x: %d" x

let nt = 256
let vt = 7
let gid = 0
let nv = nt * vt
let count = 100
let r = min nv (count - nv)

for i = 0 to 6 do
    printfn "i = %d" i