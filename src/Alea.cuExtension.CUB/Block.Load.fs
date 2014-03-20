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
    type StaticParam =
        {
            BLOCK_THREADS       : int
            ITEMS_PER_THREAD    : int
            ALGORITHM           : BlockLoadAlgorithm
            WARP_TIME_SLICING   : bool
            MAX_VEC_SIZE        : int 
            VEC_SIZE            : int 
            VECTORS_PER_THREAD  : int

            BlockExchangeParam  : BlockExchange.StaticParam
        }

        static member Init(block_threads:int, items_per_thread:int, algorithm:BlockLoadAlgorithm, warp_time_slicing:bool) =
            let max_vec_size = CUB_MIN 4 items_per_thread
            let vec_size = if ((((max_vec_size - 1) &&& max_vec_size) = 0) && ((items_per_thread % max_vec_size) = 0)) then max_vec_size else 1
            let vectors_per_thread = items_per_thread / vec_size
            let bexp = BlockExchange.StaticParam.Init(block_threads, items_per_thread, warp_time_slicing)
            {
                BLOCK_THREADS       = block_threads
                ITEMS_PER_THREAD    = items_per_thread
                ALGORITHM           = algorithm
                WARP_TIME_SLICING   = warp_time_slicing
                MAX_VEC_SIZE        = max_vec_size
                VEC_SIZE            = vec_size
                VECTORS_PER_THREAD  = vectors_per_thread

                BlockExchangeParam  = bexp
            }

        static member Init(block_threads, items_per_thread) = StaticParam.Init(block_threads, items_per_thread, BlockLoadAlgorithm.BLOCK_LOAD_DIRECT, false)


    type TempStorage<'T> = BlockExchange.TempStorage<'T>
    
    let [<ReflectedDefinition>] inline PrivateStorage<'T>(p:BlockExchange.StaticParam) = BlockExchange.PrivateStorage<'T>(p)
                           

    module LoadDirectBlocked =
        
        let [<ReflectedDefinition>] inline Default (p:StaticParam)
            (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) =
                   
            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[(linear_tid * p.ITEMS_PER_THREAD) + ITEM]

        
        let [<ReflectedDefinition>] inline Guarded (p:StaticParam) 
            (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
       
            let bounds = valid_items - (linear_tid * p.ITEMS_PER_THREAD)
            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[(linear_tid * p.ITEMS_PER_THREAD) + ITEM]

        
        let [<ReflectedDefinition>] inline GuardedWithOOB (p:StaticParam)
            (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) (oob_default:'T) =
       
            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do items.[ITEM] <- oob_default
            let bounds = valid_items - (linear_tid * p.ITEMS_PER_THREAD)
            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[(linear_tid * p.ITEMS_PER_THREAD) + ITEM]



    module LoadDirectBlockedVectorized =

        let [<ReflectedDefinition>] inline Default (p:StaticParam)
            (linear_tid:int) (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
            
            let ptr = (block_ptr + (linear_tid * p.VEC_SIZE * p.VECTORS_PER_THREAD)) |> __ptr_reinterpret

            let vec_items = __local__.Array<'T>(p.VECTORS_PER_THREAD) |> __array_to_ptr

            for ITEM = 0 to (p.VECTORS_PER_THREAD - 1) do vec_items.[ITEM] <- ptr.[ITEM]
            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do items.[ITEM] <- vec_items.[ITEM] //|> __ptr_to_obj
        

    module LoadDirectStriped =
        
        let [<ReflectedDefinition>] inline Default (p:StaticParam) (linear_tid:int)
            (block_itr:deviceptr<'T>) (items:deviceptr<'T>) =
                
            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[(ITEM * p.BLOCK_THREADS) + linear_tid]


        let [<ReflectedDefinition>] inline Guarded (p:StaticParam) (linear_tid:int)
            (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
            let bounds = valid_items - linear_tid
            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do 
                if (ITEM * p.BLOCK_THREADS < bounds) then items.[ITEM] <- block_itr.[(ITEM * p.BLOCK_THREADS) + linear_tid]


        let [<ReflectedDefinition>] inline GuardedWithOOB (p:StaticParam) (linear_tid:int)
            (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) (oob_default:'T) =
            
            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do items.[ITEM] <- oob_default
            let bounds = valid_items - linear_tid
            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do 
                if (ITEM * p.BLOCK_THREADS < bounds) then items.[ITEM] <- block_itr.[(ITEM * p.BLOCK_THREADS) + linear_tid]


    module LoadDirectWarpStriped =

        let [<ReflectedDefinition>] inline Default (p:StaticParam)
            (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) =
            let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
            let wid = linear_tid >>> CUB_PTX_LOG_WARP_THREADS
            let warp_offset = wid * CUB_PTX_WARP_THREADS * p.ITEMS_PER_THREAD

            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do items.[ITEM] <- block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)]


        let [<ReflectedDefinition>] inline Guarded (p:StaticParam) (linear_tid:int)
            (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
            
            let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
            let wid = linear_tid >>> CUB_PTX_LOG_WARP_THREADS
            let warp_offset = wid * CUB_PTX_WARP_THREADS * p.ITEMS_PER_THREAD
            let bounds = valid_items - warp_offset - tid

            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do 
                if ((ITEM * CUB_PTX_WARP_THREADS) < bounds) then items.[ITEM] <- block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)]

        
        let [<ReflectedDefinition>] inline GuardedWithOOB (p:StaticParam) (linear_tid:int)
            (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) (oob_default:'T) =
                        
            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do items.[ITEM] <- oob_default
            let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
            let wid = linear_tid >>> CUB_PTX_LOG_WARP_THREADS
            let warp_offset = wid * CUB_PTX_WARP_THREADS * p.ITEMS_PER_THREAD
            let bounds = valid_items - warp_offset - tid

            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do 
                if ((ITEM * CUB_PTX_WARP_THREADS) < bounds) then items.[ITEM] <- block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)]


    module private InternalLoad =

        module BlockLoadDirect =
        
            let [<ReflectedDefinition>] inline Default (p:StaticParam) 
                (temp_storage:TempStorage<'T>) (linear_tid:int)
                (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
                LoadDirectBlocked.Default p linear_tid block_ptr items
    

            let [<ReflectedDefinition>] inline Guarded (p:StaticParam)
                (temp_storage:TempStorage<'T>) (linear_tid:int)
                (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
                LoadDirectBlocked.Guarded p linear_tid block_ptr items valid_items
    

            let [<ReflectedDefinition>] inline GuardedWithOOB (p:StaticParam)
                (temp_storage:TempStorage<'T>) (linear_tid:int) 
                (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) (oob_default:'T) =
                LoadDirectBlocked.GuardedWithOOB p linear_tid block_ptr items valid_items oob_default
    

        module BlockLoadVectorized =
            let [<ReflectedDefinition>] inline Default (p:StaticParam)
                (temp_storage:TempStorage<'T>) (linear_tid:int)
                (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
                LoadDirectBlockedVectorized.Default p linear_tid block_ptr items
    

            let [<ReflectedDefinition>] inline Guarded (p:StaticParam)
                (temp_storage:TempStorage<'T>) (linear_tid:int)
                (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
                LoadDirectBlocked.Guarded p linear_tid block_ptr items valid_items
    
            
            let [<ReflectedDefinition>] inline GuardedWithOOB (p:StaticParam)
                (temp_storage:TempStorage<'T>) (linear_tid:int)
                (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) (oob_default:'T) =
                LoadDirectBlocked.GuardedWithOOB p linear_tid block_ptr items valid_items oob_default
    

        module BlockLoadTranspose =

            let [<ReflectedDefinition>] inline Default (p:StaticParam)
                (temp_storage:TempStorage<'T>) (linear_tid:int)
                (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
                LoadDirectStriped.Default p linear_tid block_ptr items
                BlockExchange.InstanceParam<'T>.Init(p.BlockExchangeParam, temp_storage, linear_tid).StripedToBlocked(p.BlockExchangeParam, items)
    

            let [<ReflectedDefinition>] inline Guarded (p:StaticParam)
                (temp_storage:TempStorage<'T>) (linear_tid:int)
                (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
                LoadDirectStriped.Guarded p linear_tid block_ptr items valid_items
                BlockExchange.InstanceParam<'T>.Init(p.BlockExchangeParam, temp_storage, linear_tid).StripedToBlocked(p.BlockExchangeParam, items)
    

            let [<ReflectedDefinition>] inline GuardedWithOOB (p:StaticParam)
                (temp_storage:TempStorage<'T>) (linear_tid:int)
                (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) (oob_default:'T) =
                LoadDirectStriped.GuardedWithOOB p linear_tid block_ptr items valid_items oob_default
                BlockExchange.InstanceParam<'T>.Init(p.BlockExchangeParam, temp_storage, linear_tid).StripedToBlocked(p.BlockExchangeParam, items)
    

        module BlockLoadWarpTranspose =
    
            let [<ReflectedDefinition>] inline Default (p:StaticParam)
                (temp_storage:TempStorage<'T>) (linear_tid:int)
                (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
                let WARP_THREADS = CUB_PTX_WARP_THREADS
                if (p.BLOCK_THREADS % WARP_THREADS) <> 0 then
                    ()
                else
                    LoadDirectWarpStriped.Default p linear_tid block_ptr items
                    BlockExchange.InstanceParam<'T>.Init(p.BlockExchangeParam, temp_storage, linear_tid).WarpStripedToBlocked(p.BlockExchangeParam, items)
    

            let [<ReflectedDefinition>] inline Guarded (p:StaticParam)
                (temp_storage:TempStorage<'T>) (linear_tid:int)
                (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
                let WARP_THREADS = CUB_PTX_WARP_THREADS
                if (p.BLOCK_THREADS % WARP_THREADS) <> 0 then
                    ()
                else
                    LoadDirectWarpStriped.Guarded p linear_tid block_ptr items valid_items
                    BlockExchange.InstanceParam<'T>.Init(p.BlockExchangeParam, temp_storage, linear_tid).WarpStripedToBlocked(p.BlockExchangeParam, items)
    

            let [<ReflectedDefinition>] inline GuardedWithOOB (p:StaticParam)
                (temp_storage:TempStorage<'T>) (linear_tid:int)
                (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) (oob_default:'T) =
                let WARP_THREADS = CUB_PTX_WARP_THREADS
                if (p.BLOCK_THREADS % WARP_THREADS) <> 0 then
                    ()
                else
                    LoadDirectWarpStriped.GuardedWithOOB p linear_tid block_ptr items valid_items oob_default
                    BlockExchange.InstanceParam<'T>.Init(p.BlockExchangeParam, temp_storage, linear_tid).WarpStripedToBlocked(p.BlockExchangeParam, items)
    
    
        let [<ReflectedDefinition>] inline Default (p:StaticParam) 
            (temp_storage:TempStorage<'T>) (linear_tid:int) (block_ptr:deviceptr<'T>) 
            (items:deviceptr<'T>) =
                p.ALGORITHM |> function
                | BlockLoadAlgorithm.BLOCK_LOAD_DIRECT ->          BlockLoadDirect.Default p temp_storage linear_tid block_ptr items
                | BlockLoadAlgorithm.BLOCK_LOAD_VECTORIZE ->       BlockLoadVectorized.Default p temp_storage linear_tid block_ptr items
                | BlockLoadAlgorithm.BLOCK_LOAD_TRANSPOSE ->       BlockLoadTranspose.Default p temp_storage linear_tid block_ptr items
                | BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE ->  BlockLoadWarpTranspose.Default p temp_storage linear_tid block_ptr items
                | _ -> BlockLoadDirect.Default p temp_storage linear_tid block_ptr items

        let [<ReflectedDefinition>] inline Guarded (p:StaticParam)
            (temp_storage:TempStorage<'T>) (linear_tid:int)
            (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
                p.ALGORITHM |> function
                | BlockLoadAlgorithm.BLOCK_LOAD_DIRECT ->          BlockLoadDirect.Guarded p temp_storage linear_tid block_ptr items valid_items
                | BlockLoadAlgorithm.BLOCK_LOAD_VECTORIZE ->       BlockLoadVectorized.Guarded p temp_storage linear_tid block_ptr items valid_items
                | BlockLoadAlgorithm.BLOCK_LOAD_TRANSPOSE ->       BlockLoadTranspose.Guarded p temp_storage linear_tid block_ptr items valid_items
                | BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE ->  BlockLoadWarpTranspose.Guarded p temp_storage linear_tid block_ptr items valid_items
                | _ -> BlockLoadDirect.Guarded p temp_storage linear_tid block_ptr items valid_items

        let [<ReflectedDefinition>] inline GuardedWithOOB (p:StaticParam)
            (temp_storage:TempStorage<'T>) (linear_tid:int) (block_ptr:deviceptr<'T>) 
            (items:deviceptr<'T>) (valid_items:int) (oob_default:'T) =
                p.ALGORITHM |> function
                | BlockLoadAlgorithm.BLOCK_LOAD_DIRECT ->          BlockLoadDirect.GuardedWithOOB p temp_storage linear_tid block_ptr items valid_items oob_default
                | BlockLoadAlgorithm.BLOCK_LOAD_VECTORIZE ->       BlockLoadVectorized.GuardedWithOOB p temp_storage linear_tid block_ptr items valid_items oob_default
                | BlockLoadAlgorithm.BLOCK_LOAD_TRANSPOSE ->       BlockLoadTranspose.GuardedWithOOB p temp_storage linear_tid block_ptr items valid_items oob_default
                | BlockLoadAlgorithm.BLOCK_LOAD_WARP_TRANSPOSE ->  BlockLoadWarpTranspose.GuardedWithOOB p temp_storage linear_tid block_ptr items valid_items oob_default
                | _ -> BlockLoadDirect.GuardedWithOOB p temp_storage linear_tid block_ptr items valid_items oob_default





    [<Record>]
    type InstanceParam<'T> =
        { 
            mutable temp_storage    : TempStorage<'T>
            mutable linear_tid      : int
        }

        [<ReflectedDefinition>]
        static member Init(p:StaticParam) = 
            { temp_storage = PrivateStorage<'T>(p.BlockExchangeParam); linear_tid = threadIdx.x }
        
        [<ReflectedDefinition>]
        static member Init(p:StaticParam, temp_storage:TempStorage<'T>) =
            { temp_storage = temp_storage; linear_tid = threadIdx.x }
        
        [<ReflectedDefinition>] 
        static member Init(p:StaticParam, linear_tid:int) =
            { temp_storage = PrivateStorage<'T>(p.BlockExchangeParam); linear_tid = linear_tid }
        
        [<ReflectedDefinition>]
        static member Init(p:StaticParam, temp_storage:TempStorage<'T>, linear_tid:int) =
            { temp_storage = temp_storage; linear_tid = linear_tid }
            
        [<ReflectedDefinition>] 
        member this.Load(p, block_ptr, items) = 
            InternalLoad.Default p this.temp_storage this.linear_tid block_ptr items
        
        [<ReflectedDefinition>] 
        member this.Load(p, block_ptr, items, valid_items) = 
            InternalLoad.Guarded p this.temp_storage this.linear_tid block_ptr items valid_items
        
        [<ReflectedDefinition>]
        member this.Load(p, block_ptr, items, valid_items, oob_default) = 
            InternalLoad.GuardedWithOOB p this.temp_storage this.linear_tid block_ptr items valid_items oob_default

