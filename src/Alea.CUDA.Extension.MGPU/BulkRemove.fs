module Alea.CUDA.Extension.MGPU.BulkRemove

// this maps to bulkremove.cuh

open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.Static
open Alea.CUDA.Extension.MGPU.DeviceUtil
open Alea.CUDA.Extension.MGPU.Intrinsics

(*type IBulkRemove<'Tuning, 'InputIt, 'IndiciesIt, 'OutputIt> =




let bulkRemove (inputIt:Iterator) (sourceCount:int) (indicesIt:Iterator) (indCount:int) (outputIt:Iterator) =
    
    let remove =
        <@ fun (tid:int) (x  
  
  *)