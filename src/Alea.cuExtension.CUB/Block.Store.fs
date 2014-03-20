[<AutoOpen>]
module Alea.cuExtension.CUB.Block.Store

open Microsoft.FSharp.Quotations

open Alea.CUDA
open Alea.CUDA.Utilities

open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Utilities

type BlockStoreAlgorithm =
    | BLOCK_STORE_DIRECT            = 0
    | BLOCK_STORE_VECTORIZE         = 1
    | BLOCK_STORE_TRANSPOSE         = 2
    | BLOCK_STORE_WARP_TRANSPOSE    = 3

module BlockStore =
    type StaticParam =
        {
            BLOCK_THREADS       : int
            ITEMS_PER_THREAD    : int
            ALGORITHM           : BlockStoreAlgorithm
            WARP_TIME_SLICING   : bool
            MAX_VEC_SIZE        : int
            VEC_SIZE            : int
            VECTORS_PER_THREAD  : int

            BlockExchangeParam  : BlockExchange.StaticParam
        }
                
        static member Init(block_threads, items_per_thread, algorithm, warp_time_slicing) =
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

                
            static member Init(block_threads, items_per_thread) = StaticParam.Init(block_threads, items_per_thread, BlockStoreAlgorithm.BLOCK_STORE_DIRECT, false)
        

    type TempStorage<'T> = BlockExchange.TempStorage<'T>
    
    module StoreDirectBlocked =

        let [<ReflectedDefinition>] inline Default (p:StaticParam) (linear_tid:int)
            (block_itr:deviceptr<'T>) (items:deviceptr<'T>) =
            

            for ITEM = 0 to p.ITEMS_PER_THREAD - 1 do block_itr.[(linear_tid * p.ITEMS_PER_THREAD) + ITEM] <- items.[ITEM]
        
        
        let [<ReflectedDefinition>] inline Guarded (p:StaticParam) (linear_tid:int)
            (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
            
            
            for ITEM = 0 to p.ITEMS_PER_THREAD - 1 do
                if ITEM + (linear_tid * p.ITEMS_PER_THREAD) < valid_items then
                    block_itr.[(linear_tid * p.ITEMS_PER_THREAD) + ITEM] <- items.[ITEM]
        


    module StoreDirectBlockedVectorized =
        let [<ReflectedDefinition>] inline Default (p:StaticParam) (linear_tid:int)
            (block_itr:deviceptr<'T>) (items:deviceptr<'T>) =
            ()
       
    

    module StoreDirectStriped =
        let [<ReflectedDefinition>] inline Default (p:StaticParam) (linear_tid:int)
            (block_itr:deviceptr<'T>) (items:deviceptr<'T>) =
            

            for ITEM = 0 to p.ITEMS_PER_THREAD - 1 do block_itr.[(ITEM * p.BLOCK_THREADS) + linear_tid] <- items.[ITEM]
        

        let [<ReflectedDefinition>] inline Guarded (p:StaticParam) (linear_tid:int)
            (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
            
        
            for ITEM = 0 to p.ITEMS_PER_THREAD - 1 do
                if ((ITEM * p.BLOCK_THREADS) + linear_tid) < valid_items then
                    block_itr.[(ITEM * p.BLOCK_THREADS) + linear_tid] <- items.[ITEM]
        

    module StoreDirectWarpStriped =
        let [<ReflectedDefinition>] inline Default (p:StaticParam) (linear_tid:int)
            (block_itr:deviceptr<'T>) (items:deviceptr<'T>) =
            
        
            let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
            let wid = linear_tid >>> CUB_PTX_WARP_THREADS
            let warp_offset = wid * CUB_PTX_WARP_THREADS * p.ITEMS_PER_THREAD
            
            for ITEM = 0 to p.ITEMS_PER_THREAD - 1 do block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)] <- items.[ITEM]
        
    
        let [<ReflectedDefinition>] inline Guarded (p:StaticParam) (linear_tid:int)
            (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
            
                    
            let tid = linear_tid &&& (CUB_PTX_WARP_THREADS - 1)
            let wid = linear_tid >>> CUB_PTX_WARP_THREADS
            let warp_offset = wid * CUB_PTX_WARP_THREADS * p.ITEMS_PER_THREAD
            
            for ITEM = 0 to p.ITEMS_PER_THREAD - 1 do
                if (warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS) < valid_items) then
                    block_itr.[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)] <- items.[ITEM]
        

    module InternalStore =

        module BlockStoreDirect =
        
            let [<ReflectedDefinition>] inline Default (p:StaticParam) (temp_storage:TempStorage<'T>) (linear_tid:int)
                (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
                StoreDirectBlocked.Default p linear_tid block_ptr items
            

            let [<ReflectedDefinition>] inline Guarded (p:StaticParam) (temp_storage:TempStorage<'T>) (linear_tid:int)
                (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
                StoreDirectBlocked.Guarded p linear_tid block_ptr items valid_items
  


        module BlockStoreVectorized =
            let [<ReflectedDefinition>] inline Default (p:StaticParam) (temp_storage:TempStorage<'T>) (linear_tid:int)
                (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
                StoreDirectBlockedVectorized.Default p linear_tid block_ptr items
            

            let [<ReflectedDefinition>] inline Guarded (p:StaticParam) (temp_storage:TempStorage<'T>) (linear_tid:int)
                (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
                StoreDirectBlocked.Guarded p linear_tid block_ptr items valid_items


        module BlockStoreTranspose =
        
            let [<ReflectedDefinition>] inline Default (p:StaticParam) (temp_storage:TempStorage<'T>) (linear_tid:int)
                (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
                StoreDirectStriped.Default p linear_tid block_ptr items
                BlockExchange.InstanceParam<'T>.Init(p.BlockExchangeParam, temp_storage, linear_tid).StripedToBlocked(p.BlockExchangeParam, items)
            

            let [<ReflectedDefinition>] inline Guarded (p:StaticParam) (temp_storage:TempStorage<'T>) (linear_tid:int)
                (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
                StoreDirectStriped.Guarded p linear_tid block_ptr items valid_items
                BlockExchange.InstanceParam<'T>.Init(p.BlockExchangeParam, temp_storage, linear_tid).StripedToBlocked(p.BlockExchangeParam, items)


        module BlockStoreWarpTranspose =
        
            let [<ReflectedDefinition>] inline Default (p:StaticParam) (temp_storage:TempStorage<'T>) (linear_tid:int)
                (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
                let WARP_THREADS = CUB_PTX_WARP_THREADS
                if (p.BLOCK_THREADS % WARP_THREADS) <> 0 then 
                    ()
                else
                    StoreDirectWarpStriped.Default p linear_tid block_ptr items
                    BlockExchange.InstanceParam<'T>.Init(p.BlockExchangeParam, temp_storage, linear_tid).WarpStripedToBlocked(p.BlockExchangeParam, items)
            

            let [<ReflectedDefinition>] inline Guarded (p:StaticParam) (temp_storage:TempStorage<'T>) (linear_tid:int)
                (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
                let WARP_THREADS = CUB_PTX_WARP_THREADS
                if (p.BLOCK_THREADS % WARP_THREADS) <> 0 then
                    ()
                else
                    StoreDirectWarpStriped.Guarded p linear_tid block_ptr items valid_items
                    BlockExchange.InstanceParam<'T>.Init(p.BlockExchangeParam, temp_storage, linear_tid).WarpStripedToBlocked(p.BlockExchangeParam, items)
            

        let [<ReflectedDefinition>] inline Default (p:StaticParam) (temp_storage:TempStorage<'T>) (linear_tid:int) (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) =
                p.ALGORITHM |> function
                | BlockStoreAlgorithm.BLOCK_STORE_DIRECT ->          BlockStoreDirect.Default p temp_storage linear_tid block_ptr items
                | BlockStoreAlgorithm.BLOCK_STORE_VECTORIZE ->       BlockStoreVectorized.Default p temp_storage linear_tid block_ptr items
                | BlockStoreAlgorithm.BLOCK_STORE_TRANSPOSE ->       BlockStoreTranspose.Default p temp_storage linear_tid block_ptr items
                | BlockStoreAlgorithm.BLOCK_STORE_WARP_TRANSPOSE ->  BlockStoreWarpTranspose.Default p temp_storage linear_tid block_ptr items
                | _ -> BlockStoreDirect.Default p temp_storage linear_tid block_ptr items

        let [<ReflectedDefinition>] inline Guarded (p:StaticParam) (temp_storage:TempStorage<'T>) (linear_tid:int) (block_ptr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) =
                p.ALGORITHM |> function
                | BlockStoreAlgorithm.BLOCK_STORE_DIRECT ->          BlockStoreDirect.Guarded p temp_storage linear_tid block_ptr items valid_items
                | BlockStoreAlgorithm.BLOCK_STORE_VECTORIZE ->       BlockStoreVectorized.Guarded p temp_storage linear_tid block_ptr items valid_items
                | BlockStoreAlgorithm.BLOCK_STORE_TRANSPOSE ->       BlockStoreTranspose.Guarded p temp_storage linear_tid block_ptr items valid_items
                | BlockStoreAlgorithm.BLOCK_STORE_WARP_TRANSPOSE ->  BlockStoreWarpTranspose.Guarded p temp_storage linear_tid block_ptr items valid_items
                | _ -> BlockStoreDirect.Guarded p temp_storage linear_tid block_ptr items valid_items    
    
    
    let [<ReflectedDefinition>] inline PrivateStorage<'T>(p:StaticParam) = BlockExchange.PrivateStorage<'T>(p.BlockExchangeParam)


    [<Record>]
    type InstanceParam<'T> =
        { mutable temp_storage : TempStorage<'T>; mutable linear_tid : int }

        [<ReflectedDefinition>] 
        static member Init(p:StaticParam) = 
            { temp_storage = PrivateStorage<'T>(p); linear_tid = threadIdx.x }
        
        [<ReflectedDefinition>] 
        static member Init(p:StaticParam, temp_storage:TempStorage<'T>) =
            { temp_storage = temp_storage; linear_tid = threadIdx.x }
        
        [<ReflectedDefinition>] 
        static member Init(p:StaticParam, linear_tid:int) = 
            { temp_storage = PrivateStorage<'T>(p); linear_tid = linear_tid }
        
        [<ReflectedDefinition>] 
        static member Init(p:StaticParam, temp_storage:TempStorage<'T>, linear_tid:int) = 
            { temp_storage = temp_storage; linear_tid = linear_tid }
            
        [<ReflectedDefinition>] 
        member this.Store(p, block_ptr, items) = 
            InternalStore.Default p this.temp_storage this.linear_tid block_ptr items
        
        [<ReflectedDefinition>] 
        member this.Store(p, block_ptr, items, valid_items) = 
            InternalStore.Guarded p this.temp_storage this.linear_tid block_ptr items valid_items
        