[<AutoOpen>]
module Alea.cuExtension.CUB.Block.Load
    
open Microsoft.FSharp.Quotations

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Utilities

open Macro
open Vector

type BlockLoadAlgorithm =
    | BLOCK_LOAD_DIRECT         = 0  
    | BLOCK_LOAD_VECTORIZE      = 1
    | BLOCK_LOAD_TRANSPOSE      = 2
    | BLOCK_LOAD_WARP_TRANSPOSE = 3

 

module BlockLoad =
    type Params =
        {
            BLOCK_THREADS       : int
            ITEMS_PER_THREAD    : int
            ALGORITHM           : BlockLoadAlgorithm
            WARP_TIME_SLICING   : bool
        }

        static member Init(block_threads, items_per_thread, algorithm, warp_time_slicing) =
            {
                BLOCK_THREADS       = block_threads
                ITEMS_PER_THREAD    = items_per_thread
                ALGORITHM           = algorithm
                WARP_TIME_SLICING   = warp_time_slicing
            }

                
    type Constants =
        { MAX_VEC_SIZE : int; VEC_SIZE : int; VECTORS_PER_THREAD : int }

        static member Init(p:Params) =
            let max_vec_size = CUB_MIN 4 p.ITEMS_PER_THREAD
            let vec_size = if ((((max_vec_size - 1) &&& max_vec_size) = 0) && ((p.ITEMS_PER_THREAD % max_vec_size) = 0)) then max_vec_size else 1
            let vectors_per_thread = p.ITEMS_PER_THREAD / vec_size
            { MAX_VEC_SIZE = max_vec_size; VEC_SIZE = vec_size; VECTORS_PER_THREAD = vectors_per_thread }

        
    type HostApi =
        {
            BlockExchangeHostApi: BlockExchange.HostApi
            Params              : Params
            Constants           : Constants
            SharedMemoryLength  : int 
        }

        static member Init(block_threads, items_per_thread, algorithm, warp_time_slicing) =
            let p = Params.Init(block_threads, items_per_thread, algorithm, warp_time_slicing)
            let c = Constants.Init(p)
            let bex_hApi = BlockExchange.HostApi.Init(block_threads, items_per_thread, warp_time_slicing)
            { BlockExchangeHostApi = bex_hApi; Params = p; Constants = c; SharedMemoryLength = bex_hApi.SharedMemoryLength }

        static member Init(block_threads, items_per_thread) = HostApi.Init(block_threads, items_per_thread, BlockLoadAlgorithm.BLOCK_LOAD_DIRECT, false)

    
    
    type TempStorage<'T> = BlockExchange.TempStorage<'T>
    
    let [<ReflectedDefinition>] inline PrivateStorage<'T>(h:HostApi) = BlockExchange.PrivateStorage<'T>(h.BlockExchangeHostApi)
//
//        let [<ReflectedDefinition>] inline PrivateStorage<'T>(h:Host.API) = 
//            __shared__.Array<'T>(h.SharedMemoryLength) |> __array_to_ptr
//
//        [<Record>]
//        type API<'T> =
//            { mutable temp_storage : TempStorage.API<'T>; mutable linear_tid : int }
//
////            [<ReflectedDefinition>] 
////            static member Init(h:Host.API) =
////                { temp_storage = PrivateStorage<'T>(h); linear_tid = threadIdx.x }
////
////            [<ReflectedDefinition>] 
////            static member Init(h:Host.API, temp_storage:TempStorage.API<'T>) =
////                { temp_storage = temp_storage; linear_tid = threadIdx.x }
////
////            [<ReflectedDefinition>] 
////            static member Init(h:Host.API, linear_tid:int) =
////                { temp_storage = PrivateStorage<'T>(h); linear_tid = linear_tid }
//
//            [<ReflectedDefinition>] 
//            static member Init(h:Host.API, temp_storage:TempStorage.API<'T>, linear_tid:int) =
//                { temp_storage = temp_storage; linear_tid = linear_tid }
                           

    module LoadDirectBlocked =
        open Template

        let [<ReflectedDefinition>] inline Default (h:HostApi)
            (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) =
            let p = h.Params
            
            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[(linear_tid * p.ITEMS_PER_THREAD) + ITEM]

        
        let [<ReflectedDefinition>] inline Guarded (h:HostApi) 
            (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
            let p = h.Params
            

            let bounds = valid_items - (linear_tid * p.ITEMS_PER_THREAD)
            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[(linear_tid * p.ITEMS_PER_THREAD) + ITEM]

        
        let [<ReflectedDefinition>] inline GuardedWithOOB (h:HostApi)
            (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) (oob_default:'T) =
            let p = h.Params
            

            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do items.[ITEM] <- oob_default
            let bounds = valid_items - (linear_tid * p.ITEMS_PER_THREAD)
            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[(linear_tid * p.ITEMS_PER_THREAD) + ITEM]


    //    let [<ReflectedDefinition>] inline api (h:HostApi) (linear_tid:int) = (Default h, Guarded h, GuardedWithOOB h)


    module LoadDirectBlockedVectorized =
        open Template

        let [<ReflectedDefinition>] inline Default (h:HostApi) (linear_tid:int) (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
            let p = h.Params
            let c = h.Constants            

            let ptr = (block_ptr + (linear_tid * c.VEC_SIZE * c.VECTORS_PER_THREAD)) |> __ptr_reinterpret

            let vec_items = __local__.Array<'T>(c.VECTORS_PER_THREAD) |> __array_to_ptr

            for ITEM = 0 to (c.VECTORS_PER_THREAD - 1) do vec_items.[ITEM] <- ptr.[ITEM]
            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do items.[ITEM] <- vec_items.[ITEM] //|> __ptr_to_obj
        
        //let [<ReflectedDefinition>] inline api (h:HostApi) (linear_tid:int) (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) = Default h temp_storage linear_tid block_ptr items

    module LoadDirectStriped =
        open Template

        let [<ReflectedDefinition>] inline Default (h:HostApi) (linear_tid:int)
        
                    (block_itr:deviceptr<'T>) (items:deviceptr<'T>) =
                let p = h.Params
            
                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[(ITEM * p.BLOCK_THREADS) + linear_tid]


        let [<ReflectedDefinition>] inline Guarded (h:HostApi) (linear_tid:int)
        
                    (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
                let p = h.Params
            

                let bounds = valid_items - linear_tid
                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do 
                    if (ITEM * p.BLOCK_THREADS < bounds) then items.[ITEM] <- block_itr.[(ITEM * p.BLOCK_THREADS) + linear_tid]


        let [<ReflectedDefinition>] inline GuardedWithOOB (h:HostApi) (linear_tid:int)
                (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) (oob_default:'T) =
                let p = h.Params
            

                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do items.[ITEM] <- oob_default
                let bounds = valid_items - linear_tid
                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do 
                    if (ITEM * p.BLOCK_THREADS < bounds) then items.[ITEM] <- block_itr.[(ITEM * p.BLOCK_THREADS) + linear_tid]


    //    let [<ReflectedDefinition>] inline api (h:HostApi) (linear_tid:int) = (Default h d, Guarded h d, GuardedWithOOB d)
    

    module LoadDirectWarpStriped =
        open Template

        let [<ReflectedDefinition>] inline Default (h:HostApi) (linear_tid:int)
        
                    (block_itr:deviceptr<'T>) (items:deviceptr<'T>) =
                let p = h.Params
            
            
                let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
                let wid = linear_tid >>> CUB_PTX_LOG_WARP_THREADS
                let warp_offset = wid * CUB_PTX_WARP_THREADS * p.ITEMS_PER_THREAD

                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)]


        let [<ReflectedDefinition>] inline Guarded (h:HostApi) (linear_tid:int)
        
                    (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
                let p = h.Params
            

                let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
                let wid = linear_tid >>> CUB_PTX_LOG_WARP_THREADS
                let warp_offset = wid * CUB_PTX_WARP_THREADS * p.ITEMS_PER_THREAD
                let bounds = valid_items - warp_offset - tid

                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do 
                    if ((ITEM * CUB_PTX_WARP_THREADS) < bounds) then items.[ITEM] <- block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)]

        
        let [<ReflectedDefinition>] inline GuardedWithOOB (h:HostApi) (linear_tid:int)
        
                    (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) (oob_default:'T) =
                let p = h.Params
            
            
                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do items.[ITEM] <- oob_default
                let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
                let wid = linear_tid >>> CUB_PTX_LOG_WARP_THREADS
                let warp_offset = wid * CUB_PTX_WARP_THREADS * p.ITEMS_PER_THREAD
                let bounds = valid_items - warp_offset - tid

                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do 
                    if ((ITEM * CUB_PTX_WARP_THREADS) < bounds) then items.[ITEM] <- block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)]


    //    let [<ReflectedDefinition>] inline api (h:HostApi) (linear_tid:int) = (Default h d, Guarded h d, GuardedWithOOB h d)

    module private InternalLoad =
        open Template

        module BlockLoadDirect =
        
            let [<ReflectedDefinition>] inline Default (h:HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int)
                (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
                LoadDirectBlocked.Default h linear_tid block_ptr items
    

            let [<ReflectedDefinition>] inline Guarded (h:HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int)
                (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
                LoadDirectBlocked.Guarded h linear_tid block_ptr items valid_items
    

            let [<ReflectedDefinition>] inline GuardedWithOOB (h:HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int) 
                (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) (oob_default:'T) =
                LoadDirectBlocked.GuardedWithOOB h linear_tid block_ptr items valid_items oob_default
    

    //        let [<ReflectedDefinition>] inline api (h:HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int) = (Default h d, Guarded h d, GuardedWithOOB d)


        module BlockLoadVectorized =
            let [<ReflectedDefinition>] inline Default (h:HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int)
                (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
                LoadDirectBlockedVectorized.Default h linear_tid block_ptr items
    

            let [<ReflectedDefinition>] inline Guarded (h:HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int)
                (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
                LoadDirectBlocked.Guarded h linear_tid block_ptr items valid_items
    
            
            let [<ReflectedDefinition>] inline GuardedWithOOB (h:HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int)
                (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) (oob_default:'T) =
                LoadDirectBlocked.GuardedWithOOB h linear_tid block_ptr items valid_items oob_default
    

    //        let [<ReflectedDefinition>] inline api (h:HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int) = (Default h d, Guarded h d, GuardedWithOOB d)


        module BlockLoadTranspose =
    

            let [<ReflectedDefinition>] inline Default (h:HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int)
                (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
                LoadDirectStriped.Default h linear_tid block_ptr items
                BlockExchange.API<'T>.Init(h.BlockExchangeHostApi, temp_storage, linear_tid).StripedToBlocked(h.BlockExchangeHostApi, items)
    

            let [<ReflectedDefinition>] inline Guarded (h:HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int)
                (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
                LoadDirectStriped.Guarded h linear_tid block_ptr items valid_items
                BlockExchange.API<'T>.Init(h.BlockExchangeHostApi, temp_storage, linear_tid).StripedToBlocked(h.BlockExchangeHostApi, items)
    

            let [<ReflectedDefinition>] inline GuardedWithOOB (h:HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int)
                (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) (oob_default:'T) =
                LoadDirectStriped.GuardedWithOOB h linear_tid block_ptr items valid_items oob_default
                BlockExchange.API<'T>.Init(h.BlockExchangeHostApi, temp_storage, linear_tid).StripedToBlocked(h.BlockExchangeHostApi, items)
    

    //        let [<ReflectedDefinition>] inline api (h:HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int) = (Default h d, Guarded h d, GuardedWithOOB d)


        module BlockLoadWarpTranspose =
    
            let [<ReflectedDefinition>] inline Default (h:HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int)
                (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
                let WARP_THREADS = CUB_PTX_WARP_THREADS
                if (h.Params.BLOCK_THREADS % WARP_THREADS) <> 0 then
                    ()
                else
                    LoadDirectWarpStriped.Default h linear_tid block_ptr items
                    BlockExchange.API<'T>.Init(h.BlockExchangeHostApi, temp_storage, linear_tid).WarpStripedToBlocked(h.BlockExchangeHostApi, items)
    

            let [<ReflectedDefinition>] inline Guarded (h:HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int)
                (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
                let WARP_THREADS = CUB_PTX_WARP_THREADS
                if (h.Params.BLOCK_THREADS % WARP_THREADS) <> 0 then
                    ()
                else
                    LoadDirectWarpStriped.Guarded h linear_tid block_ptr items valid_items
                    BlockExchange.API<'T>.Init(h.BlockExchangeHostApi, temp_storage, linear_tid).WarpStripedToBlocked(h.BlockExchangeHostApi, items)
    

            let [<ReflectedDefinition>] inline GuardedWithOOB (h:HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int)
                (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) (oob_default:'T) =
                let WARP_THREADS = CUB_PTX_WARP_THREADS
                if (h.Params.BLOCK_THREADS % WARP_THREADS) <> 0 then
                    ()
                else
                    LoadDirectWarpStriped.GuardedWithOOB h linear_tid block_ptr items valid_items oob_default
                    BlockExchange.API<'T>.Init(h.BlockExchangeHostApi, temp_storage, linear_tid).WarpStripedToBlocked(h.BlockExchangeHostApi, items)
    
    
        let [<ReflectedDefinition>] inline Default (h:HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int) (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
                h.Params.ALGORITHM |> function
                | BlockLoadAlgorithm.BLOCK_LOAD_DIRECT ->          BlockLoadDirect.Default h temp_storage linear_tid block_ptr items
                | BlockLoadAlgorithm.BLOCK_LOAD_VECTORIZE ->       BlockLoadVectorized.Default h temp_storage linear_tid block_ptr items
                | BlockLoadAlgorithm.BLOCK_LOAD_TRANSPOSE ->       BlockLoadTranspose.Default h temp_storage linear_tid block_ptr items
                | BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE ->  BlockLoadWarpTranspose.Default h temp_storage linear_tid block_ptr items
                | _ -> BlockLoadDirect.Default h temp_storage linear_tid block_ptr items

        let [<ReflectedDefinition>] inline Guarded (h:HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int) (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
                h.Params.ALGORITHM |> function
                | BlockLoadAlgorithm.BLOCK_LOAD_DIRECT ->          BlockLoadDirect.Guarded h temp_storage linear_tid block_ptr items valid_items
                | BlockLoadAlgorithm.BLOCK_LOAD_VECTORIZE ->       BlockLoadVectorized.Guarded h temp_storage linear_tid block_ptr items valid_items
                | BlockLoadAlgorithm.BLOCK_LOAD_TRANSPOSE ->       BlockLoadTranspose.Guarded h temp_storage linear_tid block_ptr items valid_items
                | BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE ->  BlockLoadWarpTranspose.Guarded h temp_storage linear_tid block_ptr items valid_items
                | _ -> BlockLoadDirect.Guarded h temp_storage linear_tid block_ptr items valid_items

        let [<ReflectedDefinition>] inline GuardedWithOOB (h:HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int) (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) (oob_default:'T) =
                h.Params.ALGORITHM |> function
                | BlockLoadAlgorithm.BLOCK_LOAD_DIRECT ->          BlockLoadDirect.GuardedWithOOB h temp_storage linear_tid block_ptr items valid_items oob_default
                | BlockLoadAlgorithm.BLOCK_LOAD_VECTORIZE ->       BlockLoadVectorized.GuardedWithOOB h temp_storage linear_tid block_ptr items valid_items oob_default
                | BlockLoadAlgorithm.BLOCK_LOAD_TRANSPOSE ->       BlockLoadTranspose.GuardedWithOOB h temp_storage linear_tid block_ptr items valid_items oob_default
                | BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE ->  BlockLoadWarpTranspose.GuardedWithOOB h temp_storage linear_tid block_ptr items valid_items oob_default
                | _ -> BlockLoadDirect.GuardedWithOOB h temp_storage linear_tid block_ptr items valid_items oob_default





    [<Record>]
    type API<'T> =
        { mutable temp_storage : TempStorage<'T>; mutable linear_tid : int }

        [<ReflectedDefinition>] static member Init(h:HostApi)                                                 = { temp_storage = PrivateStorage<'T>(h); linear_tid = threadIdx.x }
        [<ReflectedDefinition>] static member Init(h:HostApi, temp_storage:TempStorage<'T>)                   = { temp_storage = temp_storage; linear_tid = threadIdx.x }
        [<ReflectedDefinition>] static member Init(h:HostApi, linear_tid:int)                                 = { temp_storage = PrivateStorage<'T>(h); linear_tid = linear_tid }
        [<ReflectedDefinition>] static member Init(h:HostApi, temp_storage:TempStorage<'T>, linear_tid:int)   = { temp_storage = temp_storage; linear_tid = linear_tid }
            
        [<ReflectedDefinition>] member this.Load(h, block_ptr, items)                              = InternalLoad.Default h this.temp_storage this.linear_tid block_ptr items
        [<ReflectedDefinition>] member this.Load(h, block_ptr, items, valid_items)                 = InternalLoad.Guarded h this.temp_storage this.linear_tid block_ptr items valid_items
        [<ReflectedDefinition>] member this.Load(h, block_ptr, items, valid_items, oob_default)    = InternalLoad.GuardedWithOOB h this.temp_storage this.linear_tid block_ptr items valid_items oob_default

