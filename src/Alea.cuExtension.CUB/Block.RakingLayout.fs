[<AutoOpen>]
module Alea.cuExtension.CUB.Block.RakingLayout

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Utilities

open Macro 

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


let placementPtr (block_threads:int) (block_strips:int) =
    fun (temp_storage:deviceptr<'T>) (linear_tid:int) (block_strip:int option) ->
        let block_strip = if block_strip.IsSome then block_strip.Value else 0
        let mutable offset = (block_strip * block_threads) + linear_tid
        let SEGMENT_PADDING = (block_threads, block_strips) ||> SEGMENT_PADDING
        let SEGMENT_LENGTH = (block_threads, block_strips) ||> SEGMENT_LENGTH
        if SEGMENT_PADDING > 0 then 
            offset <- offset + offset / SEGMENT_LENGTH
        temp_storage + offset

let rakingPtr (block_threads:int) (block_strips:int) =
    fun (temp_storage:deviceptr<'T>) (linear_tid:int) ->
        let SEGMENT_LENGTH = (block_threads, block_strips) ||> SEGMENT_LENGTH
        let SEGMENT_PADDING = (block_threads, block_strips) ||> SEGMENT_PADDING
        temp_storage + (linear_tid * (SEGMENT_LENGTH + SEGMENT_PADDING))


//let blockRakingLayout block_threads block_strips =
[<Record>]
type Constants =
    {
        SHARED_ELEMENTS : int
        MAX_RAKING_THREADS : int
        SEGMENT_LENGTH : int
        RAKING_THREADS : int
        SEGMENT_PADDING : int
        GRID_ELEMENTS : int
        UNGUARDED : bool
    }

    static member Init(block_threads, block_strips) =
        {
            SHARED_ELEMENTS     = (block_threads, block_strips) ||> SHARED_ELEMENTS
            MAX_RAKING_THREADS  = (block_threads, block_strips) ||> MAX_RAKING_THREADS
            SEGMENT_LENGTH      = (block_threads, block_strips) ||> SEGMENT_LENGTH
            RAKING_THREADS      = (block_threads, block_strips) ||> RAKING_THREADS
            SEGMENT_PADDING     = (block_threads, block_strips) ||> SEGMENT_PADDING
            GRID_ELEMENTS       = (block_threads, block_strips) ||> GRID_ELEMENTS
            UNGUARDED           = (block_threads, block_strips) ||> UNGUARDED
        }

type ITempStorage<'T> = abstract temp_storage : deviceptr<'T>
let tempStorage<'T>(grid_elements:int)() = { new ITempStorage<'T> with member this.temp_storage = __shared__.Array<'T>(grid_elements) |> __array_to_ptr }

//let tempStorage<'T>() = 
//    fun grid_elements -> 
//        cuda { return! <@ fun _ -> __shared__.



[<Record>]
type BlockRakingLayout =
    {
        
        BLOCK_THREADS : int
        BLOCK_STRIPS : int
        Constants : Constants
    }

    member this.PlacementPtr = (this.BLOCK_THREADS, this.BLOCK_STRIPS) ||> placementPtr
    member this.RakingPtr = (this.BLOCK_THREADS, this.BLOCK_STRIPS) ||> rakingPtr    

    static member Init(block_threads, block_strips) =
        {
            BLOCK_THREADS = block_threads
            BLOCK_STRIPS = block_strips
            Constants = (block_threads, block_strips) |> Constants.Init
        }

    static member Init(block_threads) =
        {
            BLOCK_THREADS = block_threads
            BLOCK_STRIPS = 1
            Constants = (block_threads, 1) |> Constants.Init
        }

