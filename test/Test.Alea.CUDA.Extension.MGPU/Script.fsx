#r "System"
open System
open System.Diagnostics
let stopwatch = Stopwatch()
let rng = System.Random()

let scount = 10000000
let icount = scount / 2

let mutable d = Array.init scount (fun i -> i)
let ind = Array.init icount (fun _ -> rng.Next scount) |> Seq.distinct |> Seq.toArray |> Array.sort

let htr sd id = Set.difference (sd |> Set.ofArray) (id |> Set.ofArray) |> Set.toArray

let htr2 (sd:int[]) (id:int[]) =
    let mutable newA = Array.zeroCreate icount
    let mutable newN = 0
    for i = 0 to sd.Length - 1 do
        if (sd.[i] <> sd.[id.[newN]]) && (newN < id.Length - 1) then
            newA.[newN] <- sd.[i]
            newN <- newN + 1
    newA

stopwatch.Start()
let a1 = htr d ind
printfn "using sets: %6.3f" stopwatch.Elapsed.TotalMilliseconds
stopwatch.Stop()
stopwatch.Start()
let a2 = htr2 d ind
printfn "using 2nd method: %6.3f" stopwatch.Elapsed.TotalMilliseconds
stopwatch.Reset()

for i = 0 to icount - 1 do
    if ((a1.[i] - a2.[i]) > 1) || ((a1.[i] - a2.[i]) < (-1)) then
        printfn "POOP!!!!!!!!!"


let values n = Array.init n (fun _ -> 1)
let sizes = [0..10]

let dispBeforeScan (x:int[]) = printfn "Before Scan ==> Count: (%d),  %A" x.Length x
let dispAfterScan (x:int[]) = printfn "After Scan ==> Count: (%d),  %A" x.Length x

sizes |> Seq.iter (fun n -> values n |> dispBeforeScan
                            values n |> (Array.scan (+) 0) |> dispAfterScan )


let values2 = [| 1; 7; 4; 0; 9; 4; 8; 8; 2; 4; 5; 5; 1; 7; 1; 1; 5; 2; 7; 6 |]
dispBeforeScan values2
let r = Array.scan (+) 0 values2
dispAfterScan r

let block = 2342
let totalAtEnd = 1
let gdmx = 16
let tid = 2

let a = if (block <> 0) && (totalAtEnd <> 0) then 1 else 0
let b = if (gdmx - 1) <> 0 then 1 else 0
let c = if tid = 0 then 1 else 0
if (a = (if (b = c) then 1 else 0)) then
    printfn "TRUE"
else
    printfn "FALSE"

let ax = (block ^^^ 0) ^^^ (totalAtEnd ^^^ 0)
let bcx = ((gdmx - 1) ^^^ 0) ^^^ (tid &&& 0)
let rr = ax ^^^ bcx

//if (ax <> 0) && (bcx <> 0) then printfn "TRUE" else printfn "FALSE"

if (rr <> 0) then printfn "TRUE" else printfn "FALSE"



let hval = Array.init 10 (fun i -> i)
let hset = hval |> Set.ofArray

let hrmv = set [2; 3; 8]

let hres = Set.difference hset hrmv


let rng = System.Random()


let sourceCounts = [10e3; 50e3; 100e3; 200e3; 500e3; 1e6; 2e6; 5e6; 10e6; 20e6]
let removeCounts = [2000; 2000; 2000; 1000; 500; 400; 400; 400; 300; 300]
let brParis = List.zip sourceCounts removeCounts

let genItems sCount rCount =
    let source = Array.init sCount (fun _ -> rng.NextDouble())
    let indices = (Array.init rCount (fun _ -> rng.Next sCount)) |> Array.sort
    source, indices

let xs, xi = genItems 10 5
printfn "source: %A" xs
printfn "indices: %A" xi

let stats = [[|1.0; 2.0; 3.0; 4.0|]]

let xx = stats.[0].[1]