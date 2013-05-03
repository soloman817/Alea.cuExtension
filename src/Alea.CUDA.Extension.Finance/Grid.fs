module Alea.CUDA.Extension.Finance.Grid

open System
open Microsoft.FSharp.Quotations
open Alea.Interop.CUDA
open Alea.CUDA

open Alea.CUDA.Extension.TriDiag

open Util 

/// Simple homogeneous state grid startig at zero, covering interval [a, b]
let homogeneousGrid n a b =
    let dx = (b - a) / float(n - 1)
    let x = Array.init n (fun i -> a + float(i) * dx)
    x, dx

/// Create an exponentially grid up to tstop of step size not larger than dt, with nc condensing points in the first interval
let exponentiallyCondensedGrid tstart tstop dt nc =
    if tstart = tstop then
        [|tstart|]
    else
        let n = int(ceil (tstop-tstart)/dt)
        let dt' = (tstop-tstart) / float(n)
        let dt'' = dt' / float(1 <<< (nc+1))
        let tg1 = [0..nc] |> Seq.map (fun n -> tstart + float(1 <<< n)*dt'')
        let tg2 = [1..n] |> Seq.map (fun n -> tstart + float(n)*dt')
        Seq.concat [Seq.singleton tstart; tg1; tg2] |> Seq.toArray
        
let sinh (x : float) = Math.Sinh x
let asinh (x : float) = float(Math.Sign x) * Math.Log((Math.Abs x) + Math.Sqrt (x*x + 1.0))

/// Concentrate at critical point and map critical point to grid point
let concentratedGrid xMin xMax criticalPoint numPoints cFactor =
    if numPoints <= 1 then failwith "require 2 or more points"
    if xMin >= xMax then failwith "require xMin < xMax"
    if criticalPoint < xMin || criticalPoint > xMax then failwith "critical point must be in [xMin, xMax]"

    let alpha = (xMax - xMin) / cFactor
    let c1 = asinh((xMin - criticalPoint) / alpha)
    let c2 = asinh((xMax - criticalPoint) / alpha)

    // critical point in homogeneous y coordinates between 0 and 1
    let yCritical = -c1/(c2-c1)

    // this value should be close to zero
    if abs(alpha*sinh(c1*(1.0 - yCritical) + c2*yCritical)) > 1e-14 then failwith "consistency error"

    // build the homogeneous mesh in the xi coordinates which contains transformed critical point xiB
    let dy = 1.0 / float(numPoints - 1)
    let n = int(floor(yCritical/dy))

    let y = Array.zeroCreate numPoints
    y.[n] <- yCritical;
    for i = n-1 downto 0 do
        y.[i] <- y.[i+1] - dy
    for i = n+1 to numPoints - 1 do
        y.[i] <- y.[i-1] + dy

    Array.init numPoints (fun i -> criticalPoint + alpha * sinh(c1*(1.0 - y.[i]) + c2*y.[i]))

