#r "System"
open System
open System.Diagnostics
let stopwatch = Stopwatch()
let rng = System.Random()



let dataA' = Array.init 10 (fun i -> i)
let ins = [|5;7;9|]
let dataB' = [| 79;89;99|]
//let result : int[] = Array.zeroCreate (dataA'.Length + dataB'.Length)
//
//Array.blit dataA' 0 result 0 ins.[0]
//printfn "%A" result
//Array.set result ins.[0] dataB'.[0]
//printfn "%A" result
//
//Array.blit dataA' ins.[0] result (ins.[0] + 1) (ins.[1] - ins.[0])
//printfn "%A" result
//Array.set result (ins.[1] + 1) dataB'.[1]
//printfn "%A" result
////[|0; 1; 2; 3; 4; 79; 5; 6; 89; 0; 0; 0; 0|]
//
//Array.blit dataA' ins.[1] result (ins.[1] + 2) (ins.[2] - ins.[1])
//printfn "%A" result
//Array.set result (ins.[2] + 2) dataB'.[2]
//printfn "%A" result
//
//Array.blit dataA' ins.[2] result (ins.[2] + 3) ((ins.[2] + 3) - dataA'.Length - 1)


let hostBulkInsert (dataA:int[]) (indices:int[]) (dataB:int[]) =
    let result : int[] = Array.zeroCreate (dataA.Length + dataB.Length)
    Array.blit dataA 0 result 0 indices.[0]
    Array.set result indices.[0] dataB.[0]
    for i = 1 to indices.Length - 1 do
        Array.blit dataA indices.[i - 1] result (indices.[i - 1] + i) (indices.[i] - indices.[i - 1])
        Array.set result (indices.[i] + i) dataB.[i]
    let i = indices.Length - 1
    Array.blit dataA indices.[i] result (indices.[i] + i + 1) (result.Length - (indices.[i] + i + 1))
    result

let x = Array.init 25 (fun i -> i)
let y = [| 3; 7; 11; 14; 19 |]
let z = [| 93; 97; 911; 914; 919 |]
printfn "Result: %A" (hostBulkInsert x y z)

let w = [|2..5..100|]
let wi = [|1000..10..((w.Length*10+1000)-10)|]