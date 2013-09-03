[<AutoOpen>]
module Alea.CUDA.Extension.MGPU.Tuning

type Plan =
    {
        NT : int
        VT : int
    }

let bulkInsert = { NT = 128; VT = 7 }
let bulkRemove = { NT = 128; VT = 11 }
let intervalMove = { NT = 128; VT = 7 }
let intervalExpand = { NT = 128; VT = 7 }
let intervalGather = { NT = 128; VT = 7 }
let intervalScatter = { NT = 128; VT = 7 }
let join = { NT = 128; VT = 7 }
let loadBalance = { NT = 128; VT = 7 }
let localitySortKeys = { NT = 128; VT = 11 }
let localitySortPairs = { NT = 128; VT = 7 }
let mergeKeys = { NT = 128; VT = 11 }
let mergePairs = { NT = 128; VT = 7 }
let mergesortKeys = { NT = 256; VT = 7 }
let mergesortPairs = { NT = 256; VT = 11 }
let mergesortIndices = { NT = 256; VT = 11 }
let reduceA = { NT = 512; VT = 5 }
let reduceB = { NT = 128; VT = 9 }
let scanA = { NT = 512; VT = 3 }
let scanB = { NT = 128; VT = 7 }
// let search = N/A
//let segmentedSort =
//let sets =
let sortedSearch ={ NT = 128; VT = 7 }

type KernelType =
    | BulkInsert
    | BulkRemove
    | IntervalMove
    | IntervalExpand
    | IntervalGather
    | IntervalScatter
    | Join
    | LoadBalance
    | LocalitySortKeys
    | LocalitySortPairs
    | MergeKeys
    | MergePairs
    | MergesortKeys
    | MergesortPairs
    | MergesortIndices
    | ReduceA
    | ReduceB
    | ScanA
    | ScanB
//    | Search
//    | SegmentedSort
//    | Sets
    | SortedSearch


let defaultPlan (kt:KernelType) =
    match kt with
    | BulkInsert -> bulkInsert
    | BulkRemove -> bulkRemove
    | IntervalMove -> intervalMove
    | IntervalExpand -> intervalExpand
    | IntervalGather -> intervalGather
    | IntervalScatter -> intervalScatter
    | Join -> join
    | LoadBalance  -> loadBalance
    | LocalitySortKeys -> localitySortKeys
    | LocalitySortPairs -> localitySortPairs 
    | MergeKeys -> mergeKeys
    | MergePairs -> mergePairs
    | MergesortKeys -> mergesortKeys
    | MergesortPairs -> mergesortPairs
    | MergesortIndices  -> mergesortIndices
    | ReduceA -> reduceA
    | ReduceB -> reduceB
    | ScanA -> scanA
    | ScanB -> scanB
//    | Search
//    | SegmentedSort
//    | Sets
    | SortedSearch -> sortedSearch