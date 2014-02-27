[<AutoOpen>]
module Alea.cuExtension.CUB.Block.RakingLayout

open Microsoft.FSharp.Quotations

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Utilities

open Macro 
//
//[<RequireQualifiedAccess>]
//type TempStorage = deviceptr<int>
//
//let TempStorage grid_elements = __shared__.Array(grid_elements)
module TempStorage =
    type API = deviceptr<int>

    let initialize grid_elements =
        API((__shared__.Array(grid_elements) |> __array_to_ptr).Handle)
        


module private Internal =
    type TempStorage = TempStorage.API

    module Constants =
        let SHARED_ELEMENTS =
            fun block_threads block_strips ->
                block_threads * block_strips

        let MAX_RAKING_THREADS =
            fun block_threads _ ->
                CUB_MIN block_threads CUB_PTX_WARP_THREADS

        let SEGMENT_LENGTH = 
            fun block_threads block_strips ->
                let SHARED_ELEMENTS = (block_threads, block_strips) ||> SHARED_ELEMENTS
                let MAX_RAKING_THREADS = (block_threads, block_strips) ||> MAX_RAKING_THREADS
                (SHARED_ELEMENTS + MAX_RAKING_THREADS - 1) / MAX_RAKING_THREADS

        let RAKING_THREADS =
            fun block_threads block_strips ->
                let SHARED_ELEMENTS = (block_threads, block_strips) ||> SHARED_ELEMENTS
                let SEGMENT_LENGTH = (block_threads, block_strips) ||> SEGMENT_LENGTH
                (SHARED_ELEMENTS + SEGMENT_LENGTH - 1) / SEGMENT_LENGTH

 
        let SEGMENT_PADDING =
            fun block_threads block_strips ->
                let SEGMENT_LENGTH = (block_threads, block_strips) ||> SEGMENT_LENGTH
                if (CUB_PTX_SMEM_BANKS % SEGMENT_LENGTH = 0) then 1 else 0

        let GRID_ELEMENTS =
            fun block_threads block_strips ->
                let RAKING_THREADS  = (block_threads, block_strips) ||> RAKING_THREADS
                let SEGMENT_LENGTH  = (block_threads, block_strips) ||> SEGMENT_LENGTH
                let SEGMENT_PADDING = (block_threads, block_strips) ||> SEGMENT_PADDING
                RAKING_THREADS * (SEGMENT_LENGTH + SEGMENT_PADDING)

        let UNGUARDED =
            fun block_threads block_strips ->
                let SHARED_ELEMENTS = (block_threads, block_strips) ||> SHARED_ELEMENTS
                let RAKING_THREADS = (block_threads, block_strips) ||> RAKING_THREADS
                (SHARED_ELEMENTS % RAKING_THREADS = 0)
    
    module Sig =
        module PlacementPtr =
            type DefaultExpr = Expr<TempStorage -> int -> deviceptr<int>>
            type WithBlockStripsExpr = Expr<TempStorage -> int -> int -> deviceptr<int>>

        module RakingPtr =
            type DefaultExpr = Expr<TempStorage -> int -> deviceptr<int>>



module PlacementPtr =
    open Internal

    type API =
        {
            Default         : Sig.PlacementPtr.DefaultExpr
            WithBlockStrips : Sig.PlacementPtr.WithBlockStripsExpr
        }

    let private WithBlockStrips block_threads block_strips =
        let SEGMENT_PADDING = (block_threads, block_strips) ||> Constants.SEGMENT_PADDING
        let SEGMENT_LENGTH  = (block_threads, block_strips) ||> Constants.SEGMENT_LENGTH
        <@ fun (temp_storage:TempStorage) (linear_tid:int) (block_strip:int) ->
            let mutable offset = (block_strip * block_threads) + linear_tid
            if SEGMENT_PADDING > 0 then offset <- offset + offset / SEGMENT_LENGTH

            temp_storage + offset
        @>

    let private Default block_threads block_strips =
        let block_strip = 0
        let WithBlockStrips = (block_threads, block_strips) ||> WithBlockStrips
        <@ fun (temp_storage:TempStorage) (linear_tid:int) ->
            (temp_storage, linear_tid, block_strip) |||> %WithBlockStrips
        @>

    let api block_threads block_strips =
        {
            Default         =   Default
                                <|| (block_threads, block_strips)

            WithBlockStrips =   WithBlockStrips
                                <|| (block_threads, block_strips)
        }


module RakingPtr =
    open Internal

    type API =
        {
            Default : Sig.RakingPtr.DefaultExpr
        }

    let private Default block_threads block_strips =
        let SEGMENT_LENGTH  = (block_threads, block_strips) ||> Constants.SEGMENT_LENGTH
        let SEGMENT_PADDING = (block_threads, block_strips) ||> Constants.SEGMENT_PADDING

        <@ fun (temp_storage:TempStorage) (linear_tid:int) ->
            temp_storage + (linear_tid * (SEGMENT_LENGTH + SEGMENT_PADDING))
        @>

    let api block_threads block_strips =
        {
            Default =   Default
                        <|| (block_threads, block_strips)
        }


module BlockRakingLayout =
    open Internal


    type Constants =
        {
            SHARED_ELEMENTS     :   int
            MAX_RAKING_THREADS  :   int
            SEGMENT_LENGTH      :   int
            RAKING_THREADS      :   int
            SEGMENT_PADDING     :   int
            GRID_ELEMENTS       :   int
            UNGUARDED           :   int
        }

    let private init block_threads block_strips =
        {
            SHARED_ELEMENTS     = (block_threads, block_strips) ||> Internal.Constants.SHARED_ELEMENTS
            MAX_RAKING_THREADS  = (block_threads, block_strips) ||> Internal.Constants.MAX_RAKING_THREADS
            SEGMENT_LENGTH      = (block_threads, block_strips) ||> Internal.Constants.SEGMENT_LENGTH
            RAKING_THREADS      = (block_threads, block_strips) ||> Internal.Constants.RAKING_THREADS
            SEGMENT_PADDING     = (block_threads, block_strips) ||> Internal.Constants.SEGMENT_PADDING
            GRID_ELEMENTS       = (block_threads, block_strips) ||> Internal.Constants.GRID_ELEMENTS
            UNGUARDED           = (block_threads, block_strips) ||> Internal.Constants.SHARED_ELEMENTS
        }

    type API = 
        {
            Constants       : Constants
            PlacementPtr    : PlacementPtr.API
            RakingPtr       : RakingPtr.API
        }

    let api block_threads block_strips =
        block_strips |> function
        | None ->
            {
                Constants       =   (block_threads, 1) ||> init

                PlacementPtr    =   PlacementPtr.api
                                    <|| (block_threads, 1)

                RakingPtr       =   RakingPtr.api
                                    <|| (block_threads, 1)
            }

        | Some block_strips ->
            {
                Constants       =   (block_threads, block_strips) ||> init

                PlacementPtr    =   PlacementPtr.api
                                    <|| (block_threads, block_strips)

                RakingPtr       =   RakingPtr.api
                                    <|| (block_threads, block_strips)
            }


//
//let inline placementPtr (block_threads:int) (block_strips:int) =
//    fun (temp_storage:deviceptr<int>) (linear_tid:int) (block_strip:int option) ->
//        let block_strip = if block_strip.IsSome then block_strip.Value else 0
//        let mutable offset = (block_strip * block_threads) + linear_tid
//        let SEGMENT_PADDING = (block_threads, block_strips) ||> SEGMENT_PADDING
//        let SEGMENT_LENGTH = (block_threads, block_strips) ||> SEGMENT_LENGTH
//        if SEGMENT_PADDING > 0 then 
//            offset <- offset + offset / SEGMENT_LENGTH
//        temp_storage + offset
//
//let inline rakingPtr (block_threads:int) (block_strips:int) =
//    fun (temp_storage:deviceptr<int>) (linear_tid:int) ->
//        let SEGMENT_LENGTH = (block_threads, block_strips) ||> SEGMENT_LENGTH
//        let SEGMENT_PADDING = (block_threads, block_strips) ||> SEGMENT_PADDING
//        temp_storage + (linear_tid * (SEGMENT_LENGTH + SEGMENT_PADDING))
//

////let blockRakingLayout block_threads block_strips =
//[<Record>]
//type Constants =
//    {
//        SHARED_ELEMENTS : int
//        MAX_RAKING_THREADS : int
//        SEGMENT_LENGTH : int
//        RAKING_THREADS : int
//        SEGMENT_PADDING : int
//        GRID_ELEMENTS : int
//        UNGUARDED : bool
//    }
//
//    static member Init(block_threads, block_strips) =
//        {
//            SHARED_ELEMENTS     = (block_threads, block_strips) ||> SHARED_ELEMENTS
//            MAX_RAKING_THREADS  = (block_threads, block_strips) ||> MAX_RAKING_THREADS
//            SEGMENT_LENGTH      = (block_threads, block_strips) ||> SEGMENT_LENGTH
//            RAKING_THREADS      = (block_threads, block_strips) ||> RAKING_THREADS
//            SEGMENT_PADDING     = (block_threads, block_strips) ||> SEGMENT_PADDING
//            GRID_ELEMENTS       = (block_threads, block_strips) ||> GRID_ELEMENTS
//            UNGUARDED           = (block_threads, block_strips) ||> UNGUARDED
//        }
//
//type ITempStorage = abstract temp_storage : deviceptr<int>
////let tempStorage(grid_elements:int)() = { new ITempStorage<int> with member this.temp_storage = __shared__.Array(grid_elements) |> __array_to_ptr }
//let tempStorage(grid_elements:int)() = __shared__.Array(grid_elements) |> __array_to_ptr
//
////let tempStorage() = 
////    fun grid_elements -> 
////        cuda { return! <@ fun _ -> __shared__.
//
//
//
//[<Record>]
//type BlockRakingLayout =
//    {
//        
//        BLOCK_THREADS : int
//        BLOCK_STRIPS : int
//        Constants : Constants
//    }
//
//    member this.PlacementPtr = (this.BLOCK_THREADS, this.BLOCK_STRIPS) ||> placementPtr
//    member this.RakingPtr = (this.BLOCK_THREADS, this.BLOCK_STRIPS) ||> rakingPtr   
//
//    static member Init(block_threads, block_strips) =
//        {
//            BLOCK_THREADS = block_threads
//            BLOCK_STRIPS = block_strips
//            Constants = (block_threads, block_strips) |> Constants.Init
//        }
//
//    static member Init(block_threads) =
//        {
//            BLOCK_THREADS = block_threads
//            BLOCK_STRIPS = 1
//            Constants = (block_threads, 1) |> Constants.Init
//        }
//
