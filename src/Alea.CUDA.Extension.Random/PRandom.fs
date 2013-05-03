module Alea.CUDA.Extension.Random.PRandom

open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Extension

// %XIANG% (3)

// in the raw impl (in XorShift7.fs), we work with raw pointers
// but now here in the high level wrapper, I will give some high
// level struct which gives more information, and use those
// delayed resource such as DArray, DMatrix.
type XorShift7RandomNumber<'T when 'T:unmanaged> =
    {
        NumStreams : int
        NumSteps : int
        Numbers : DArray<'T>
    }

type XorShift7StartState =
    {
        Seed : uint32 option
        StartState : DArray<uint32>
    }

type XorShift7Rng<'T when 'T:unmanaged> =
    {
        StreamUnit : int
        MallocStartStateInBlob : unit -> PCalc<XorShift7StartState>
        UpdateStartState : XorShift7StartState -> uint32 -> PCalc<XorShift7StartState>
        MallocRandomNumberInBlob : int -> int -> PCalc<XorShift7RandomNumber<'T>>
        UpdateRandomNumber : XorShift7RandomNumber<'T> -> XorShift7StartState -> int -> int -> PCalc<unit>
    }

let xorshift7Rng converter = cuda {
    let! generator = XorShift7.generator converter

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let generator = generator.Apply m
        let jumpAheadMatrices = worker.Malloc(generator.JumpAheadMatrices)

        let mallocStartStateInBlob () = pcalc {
            let! startState = DArray.createInBlob worker 8
            let startState = { Seed = None; StartState = startState }
            return startState }

        let updateStartState (startState:XorShift7StartState) (seed:uint32) = pcalc {
            do!
                fun hint ->
                    let startStateHost = generator.GenerateStartState seed
                    DevicePtrUtil.Scatter(worker, startStateHost, startState.StartState.Ptr, startStateHost.Length)
                |> PCalc.action 
            return { startState with Seed = Some seed } }

        let mallocRandomNumberInBlob streams steps = pcalc {
            let! numbers = DArray.createInBlob<'T> worker (streams * steps)
            return { NumStreams = streams; NumSteps = steps; Numbers = numbers } }

        let updateRandomNumber (randomNumber:XorShift7RandomNumber<'T>) (startState:XorShift7StartState) runs rank = pcalc {
            do!
                fun hint ->
                    generator.Generate hint jumpAheadMatrices.Ptr randomNumber.NumStreams randomNumber.NumSteps startState.StartState.Ptr runs rank randomNumber.Numbers.Ptr
                |> PCalc.action }

        { StreamUnit = generator.StreamUnit
          MallocStartStateInBlob = mallocStartStateInBlob
          UpdateStartState = updateStartState
          MallocRandomNumberInBlob = mallocRandomNumberInBlob
          UpdateRandomNumber = updateRandomNumber } ) }

let xorshift7 converter = cuda {
    let! rng = xorshift7Rng converter

    return PFunc(fun (m:Module) ->
        let rng = rng.Apply m
        fun streams steps seed runs rank ->
            pcalc {
                let! s0 = rng.MallocStartStateInBlob()
                let! s1 = rng.UpdateStartState s0 seed
                let! rn = rng.MallocRandomNumberInBlob streams steps
                do! rng.UpdateRandomNumber rn s1 runs rank
                return rn } ) }

let sobol converter = cuda {
    let! generator = Sobol.generator converter

    return PFunc(fun (m:Module) ->
        let generator = generator.Apply m
        let worker = m.Worker
        fun dimensions vectors offset ->
            let generator = generator dimensions
            pcalc {
                let! directions = DArray.scatterInBlob worker generator.Directions
                let! output = DArray.createInBlob worker (dimensions * vectors)
                do! PCalc.action (fun hint -> generator.Generate hint vectors offset directions.Ptr output.Ptr)
                return output } ) }

let sobolRng converter = cuda {
    let! generator = Sobol.generator converter

    return PFunc(fun (m:Module) ->
        let generator = generator.Apply m
        let worker = m.Worker
        fun dimensions ->
            let generator = generator dimensions
            pcalc {
                let! directions = DArray.scatterInBlob worker generator.Directions
                return fun vectors offset (output:DArray<'T>) ->
                    pcalc { do! PCalc.action (fun hint -> generator.Generate hint vectors offset directions.Ptr output.Ptr) } } ) }
