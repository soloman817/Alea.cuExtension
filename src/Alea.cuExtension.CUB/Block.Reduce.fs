[<AutoOpen>]
module Alea.cuExtension.CUB.Block.Reduce

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common
//
//type BlockReduceAlgorithm =
//    | BLOCK_REDUCE_RAKING
//    | BLOCK_REDUCE_WARP_REDUCTIONS
//
//
//
//[<Record>]
//type ThreadFields<'T> =
//    {
//        temp_storage : deviceptr<'T>
//        linear_tid : int
//    }
//
//    member this.Get() = (this.temp_storage, this.linear_tid)
//
//    static member Init() =        
//        {
//            temp_storage = privateStorage()
//            linear_tid = threadIdx.x
//        }
//
//    static member inline Init(temp_storage:deviceptr<_>) =
//        {
//            temp_storage = temp_storage
//            linear_tid = threadIdx.x
//        }
//
//    static member Init(linear_tid:int) =
//        {
//            temp_storage = privateStorage()
//            linear_tid = linear_tid
//        }
//
//    static member inline Init(temp_storage:deviceptr<_>, linear_tid:int) =
//        {
//            temp_storage = temp_storage
//            linear_tid = linear_tid
//        }
//
//
////let internalBlockReduce ()
//
//
//[<Record>]
//type BlockReduce<'T> =
//    {
//        ThreadFields : ThreadFields<'T>
//        BLOCK_THREADS : int
//        ALGORITHM : BlockReduceAlgorithm
//    }

