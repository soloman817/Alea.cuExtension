[<AutoOpen>]
module Alea.cuExtension.CUB.Block.RakingLayout

open Microsoft.FSharp.Quotations

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Utilities

open Macro 

module Template =
    [<AutoOpen>]
    module Params =
        [<Record>]
        type API<'T> =
            {
                BLOCK_THREADS   : int
                BLOCK_STRIPS    : int
            }

            [<ReflectedDefinition>]
            member this.Get() = (this.BLOCK_THREADS, this.BLOCK_STRIPS)
            
            [<ReflectedDefinition>]
            static member Init<'T>(block_threads, block_strips) : API<'T> = {BLOCK_THREADS = block_threads; BLOCK_STRIPS = block_strips}

            [<ReflectedDefinition>]
            static member Default<'T>(block_threads) = API<'T>.Init(block_threads, 1)
            
    [<AutoOpen>]
    module Constants =
        type API =
            {
                SHARED_ELEMENTS     : int
                MAX_RAKING_THREADS  : int
                SEGMENT_LENGTH      : int
                RAKING_THREADS      : int
                SEGMENT_PADDING     : int
                GRID_ELEMENTS       : int
                UNGUARDED           : bool
            }

            static member Init(block_threads, block_strips) =
                let shared_elements     = block_threads * block_strips
                let max_raking_threads  = CUB_MIN block_threads CUB_PTX_WARP_THREADS
                let segment_length      = (shared_elements + max_raking_threads - 1) / max_raking_threads
                let raking_threads      = (shared_elements + segment_length - 1) / segment_length
                let segment_padding     = if (CUB_PTX_SMEM_BANKS % segment_length = 0) then 1 else 0
                let grid_elements       = raking_threads * (segment_length + segment_padding)
                let unguarded           = (shared_elements % raking_threads = 0)
                {
                    SHARED_ELEMENTS     = shared_elements
                    MAX_RAKING_THREADS  = max_raking_threads
                    SEGMENT_LENGTH      = max_raking_threads
                    RAKING_THREADS      = raking_threads
                    SEGMENT_PADDING     = segment_padding
                    GRID_ELEMENTS       = grid_elements
                    UNGUARDED           = unguarded
                }

            static member Init(tp:Params.API) = API.Init(tp.BLOCK_THREADS, template.BLOCK_STRIPS)
    
       
    [<AutoOpen>]
    module TempStorage =
        [<Record>]
        type API<'T> =
            {
                mutable Ptr         : deviceptr<'T>
                mutable Length      : int
            }

            member this.Item
                with    [<ReflectedDefinition>] get (idx:int) = this.Ptr.[idx]
                and     [<ReflectedDefinition>] set (idx:int) (v:'T) = this.Ptr.[idx] <- v

            [<ReflectedDefinition>]
            static member Init(grid_elements) =
                let s = __shared__.Array(grid_elements)
                let ptr = s |> __array_to_ptr
                { Ptr = ptr; Length = grid_elements}


    type _TemplateParams<'T>    = Params.API
    type _Constants             = Constants.API
    type _TempStorage<'T>       = TempStorage.API<'T>


module private Internal =
    open Template

    module Sig =
        module PlacementPtr =
            type Default<'T>            = _TempStorage<'T> -> int -> deviceptr<'T>
            type WithBlockStrips<'T>    = _TempStorage<'T> -> int -> int -> deviceptr<'T>

        module RakingPtr =
            type Default<'T> = _TempStorage<'T> -> int -> deviceptr<'T>



module PlacementPtr =
    open Template
    open Internal

    [<Record>]
    type API<'T> =
        {
            Default         : Sig.PlacementPtr.Default<'T>
            WithBlockStrips : Sig.PlacementPtr.WithBlockStrips<'T>
        }

    let [<ReflectedDefinition>] inline WithBlockStrips (template:_Template<'T>)
        (temp_storage:_TempStorage<'T>) (linear_tid:int) (block_strip:int) =
        let c = _Constants.Init(tp)
        let mutable offset = (block_strip * template.BLOCK_THREADS) + linear_tid
        if c.SEGMENT_PADDING > 0 then offset <- offset + offset / c.SEGMENT_LENGTH

        temp_storage.Ptr + offset
    

    let [<ReflectedDefinition>] inline Default (template:_Template<'T>)
        (temp_storage:_TempStorage<'T>) (linear_tid:int) =
        let block_strip = 0
        WithBlockStrips template temp_storage linear_tid block_strip
        

    let [<ReflectedDefinition>] api (template:_Template<'T>)=
        {
            Default         =   Default template
            WithBlockStrips =   WithBlockStrips template
        }


module RakingPtr =
    open Template
    open Internal

    [<Record>]
    type API<'T> =
        {
            Default : Sig.RakingPtr.Default<'T>
        }

    let [<ReflectedDefinition>] inline Default (template:_Template<'T>)
        (temp_storage:_TempStorage<'T>) (linear_tid:int) =
        let c = _Constants.Init template
        temp_storage.Ptr + (linear_tid * (c.SEGMENT_LENGTH + c.SEGMENT_PADDING))
        
    let [<ReflectedDefinition>] api (template:_Template<'T>) =
        {
            Default =   Default template
        }


module BlockRakingLayout =
    open Template
    open Internal

    [<Record>]
    type API<'T> =
        {
            Constants       : _Constants
            TempStorage     : _TempStorage<'T>
            PlacementPtr    : PlacementPtr.API<'T>
            RakingPtr       : RakingPtr.API<'T>
        }

        [<ReflectedDefinition>]
        static member Init(block_threads, block_strips) =
            let template = _TemplateParams.Init(block_threads, block_strips)
            let c = _Constants.Init template
            {
                Constants       = c
                TempStorage     = _TempStorage<'T>.Init c.GRID_ELEMENTS
                PlacementPtr    = PlacementPtr.api template
                RakingPtr       = RakingPtr.api template
            }

        [<ReflectedDefinition>]
        static member Init(block_threads) =
            let template = _TemplateParams.Default(block_threads)
            let c = _Constants.Init template
            {
                Constants       = c
                TempStorage     = _TempStorage<'T>.Init c.GRID_ELEMENTS
                PlacementPtr    = PlacementPtr.api template
                RakingPtr       = RakingPtr.api template
            }


    let [<ReflectedDefinition>] api (template:_Template<'T>) =
        let c = _Constants.Init template
        {
            Constants       = c
            TempStorage     = _TempStorage<'T>.Init c.GRID_ELEMENTS
            PlacementPtr    = PlacementPtr.api template
            RakingPtr       = RakingPtr.api template
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
