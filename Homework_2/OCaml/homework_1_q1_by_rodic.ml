open Lacaml.D
open Lacaml.Io
open Format

let rows = 1000   (* num of coins *)
let cols = 10     (* num of flips *)
let exps = 100000 (* num of experiments *)

let _ = Random.self_init ()

let bool_of_float f = if f >= 0. then 1. else 0.

let find_nus () =
  let flips = Mat.map bool_of_float (Mat.random rows cols) in
  let heads = Mat.as_vec (gemm flips (Mat.make_mvec cols 1.)) in
  let first = heads.{1}
  and min   = Vec.min heads
  and rand  = heads.{1 + Random.int rows} in 
  Vec.of_array[| first; min; rand |]

let rec find_avg acc runs =
  if runs == 0
  then Vec.map (fun x -> x /. float_of_int(exps * cols)) acc
  else find_avg (Vec.add acc (find_nus ())) (runs-1)

let _ = printf "@\n%a\n" pp_vec (find_avg  (Vec.make0 3) exps)