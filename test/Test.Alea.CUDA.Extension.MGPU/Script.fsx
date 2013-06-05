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
