module Test.Alea.CUDA.Extension.MGPU.BulkInsert

open System
open System.Diagnostics
open System.Collections.Generic
open Microsoft.FSharp.Quotations
open Microsoft.FSharp.Quotations.DerivedPatterns
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.BulkRemove
open Test.Alea.CUDA.Extension.MGPU.Util
open Test.Alea.CUDA.Extension.MGPU.BenchmarkStats

open NUnit.Framework

let worker = Engine.workers.DefaultWorker
let rng = System.Random()

let sourceCounts = BenchmarkStats.sourceCounts
let nIterations = BenchmarkStats.bulkRemoveIterations

[<Test>]
let ``bulkInsert moderngpu website example 1`` () =
    let hValues = Array.init 100 int

    let hValues2 = [|   0;  1; 1000;  2;     3;  4;    5;  6; 1010;  7;
                        8;  9;   10; 11;  1020; 12;   13; 14;   15; 16;
                     1030; 17;   18; 19;    20; 21; 1040; 22;   23; 24;
                       25; 26; 1050; 27;    28; 29;   30; 31; 1060; 32;
                       33; 34;   35; 36;  1070; 37;   38; 39;   40; 41;
                     1080; 42;   43; 44;    45; 46; 1090; 47;   48; 49;
                       50; 51; 1100; 52;    53; 54;   55; 56; 1110; 57;
                       58; 59;   60; 61;  1120; 62;   63; 64;   65; 66;
                     1130; 67;   68; 69;    70; 71; 1140; 72;   73; 74;
                       75; 76; 1150; 77;    78; 79;   80; 81; 1160; 82;
                       83; 84;   85; 86;  1170; 87;   88; 89;   90; 91;
                     1180; 92;   93; 94;    95; 96; 1190; 97;   98; 99 |]

    printfn "derp"


[<Test>]
let ``bulkInsert moderngpu website example 2`` () =
    let hValues = Array.init 400 int

    let insertionIndices = [|   1;   12;   13;   14;   14;   18;   20;   38;   39;   44;
                               45;   50;   50;   50;   54;   56;   59;   63;   68;   69;
                               74;   75;   84;   84;   88;  111;  111;  119;  121;  123;
                              126;  127;  144;  153;  157;  159;  163;  169;  169;  175;
                              178;  183;  190;  194;  195;  196;  196;  201;  219;  219;
                              253;  256;  259;  262;  262;  266;  272;  273;  278;  283;
                              284;  291;  296;  297;  302;  303;  306;  306;  317;  318;
                              318;  319;  319;  320;  320;  323;  326;  329;  330;  334;
                              340;  349;  352;  363;  366;  367;  369;  374;  381;  383;
                              383;  384;  386;  388;  388;  389;  393;  398;  398;  399 |]

    

