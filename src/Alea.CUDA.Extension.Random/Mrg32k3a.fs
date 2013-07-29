module Alea.CUDA.Extension.Random.Mrg32k3a

open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Extension


type XorShift7Rng<'T when 'T:unmanaged> =
    {
        StreamUnit : int
        JumpAheadMatrices : uint32[]
        GenerateStartState : uint32 -> uint32[]
        // hint -> jumpAheadMatrices -> streams -> steps -> startState -> runs -> rank -> result
        Generate : ActionHint -> DevicePtr<uint32> -> int -> int -> DevicePtr<uint32> -> int -> int -> DevicePtr<'T> -> unit
    }


/// Mrg32ka is based on the following papers
///
/// Good parameter sets for combined multiple recursive random number generators  
/// P. L'Ecuyer. Operations Research, 47(1):159-164, 1999.                        
/// http://www.iro.umontreal.ca/~lecuyer/myftp/papers/combmrg2.ps                 
///                                                                               
/// An object-oriented random-number package with many long streams and substreams
/// P. L'Ecuyer, R. Simar, E.J. Chen and W.D. Kelton.                             
/// Operations Research, 50(6):1073-1075, 2002                                    
/// http://www.iro.umontreal.ca/~lecuyer/myftp/papers/streams00.pdf 

let [<ReflectedDefinition>] mrg32k3a_pi = 3.14159265358979323846
let [<ReflectedDefinition>] mrg32k3a_norm = 2.3283065498378288e-10
let [<ReflectedDefinition>] mrg32k3a_2pow32m1 = 4294967295ul
let [<ReflectedDefinition>] mrg32k3a_m1 = 4294967087ul
let [<ReflectedDefinition>] mrg32k3a_m1c = 209ul
let [<ReflectedDefinition>] mrg32k3a_m2 = 4294944443ul
let [<ReflectedDefinition>] mrg32k3a_m2c = 22853ul
let [<ReflectedDefinition>] mrg32k3a_a12 = 1403580ul
let [<ReflectedDefinition>] mrg32k3a_a13 = (4294967087ul -  810728ul)
let [<ReflectedDefinition>] mrg32k3a_a13n = 810728ul
let [<ReflectedDefinition>] mrg32k3a_a21 = 527612UL
let [<ReflectedDefinition>] mrg32k3a_a23 = (4294944443UL - 1370589UL)
let [<ReflectedDefinition>] mrg32k3a_a23n = 1370589UL

    
//let mrg32k3a = cuda {
//
//    let mrg32k3a_v1 = defineConstantArray<uint32>(3)
//    let mrg32k3a_A1 = defineConstantArray<uint32>(9*148)
//    let mrg32k3a_v2 = defineConstantArray<uint32>(3)
//    let mrg32k3a_A2 = defineConstantArray<uint32>(9*148) 
//
//            
//            }
