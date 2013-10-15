open Lacaml.D
open Lacaml.Io
open Format

(* Homework 2 

 * Learning from Data, Edx, fall 2013
 * author: Aleksandar Rodic

 * depends on: https://bitbucket.org/mmottl/lacaml

 *)


(* Helpers *)

let _ = Random.self_init ()

let classify ws pt =
  if dot ws pt >= 0. then 1. else (-1.)

let target () =
  let p1 = Vec.random 2
  and p2 = Vec.random 2                             in
  let m  = (p1.{2} -. p2.{2}) /. (p1.{1} -. p2.{1}) in
  let b  = p1.{2} -. p1.{1} *. m                    in
  let ws = Vec.of_array [| b; m; -1. |]             in
  classify ws

let get_labels dataset classifier =
  let rows = Mat.dim1 dataset in
  Vec.init rows (fun i -> classifier (Mat.copy_row dataset i))

let pinv x =
  let xtx  = gemm (Mat.transpose x) x in
  let _    = getri xtx                in 
  gemm xtx (Mat.transpose x)

let solve x y =
  Mat.as_vec (gemm (pinv x) (Mat.from_col_vec y))

let create_dataset ?(cols = 3) rows =
  (* rows: num of points; cols: coords where the first coord is 1. *)
  Mat.map (fun _ -> 1.) ~b:(Mat.random rows cols) (Mat.create rows 1)

let hypo_error labels dataset hypo =
  let labels' = get_labels dataset hypo in
  let err_vec = Vec.mul labels labels'
  and cnt c x = 
    if x < 0. then c +. 1. else c       in
  Vec.fold cnt 0. err_vec /. float_of_int (Mat.dim1 dataset)

let avg ?(runs=1000) func =
  let rec aux run acc =
    if run == 0
    then acc /. float_of_int runs
    else aux (run-1) (acc +. func ()) in
  aux runs 0.


(* Question 5 *)

let e_in () =
  let learnset = create_dataset 100                in
  let labels   = get_labels learnset (target ())   in
  let hypo     = solve learnset labels |> classify in
  hypo_error labels learnset hypo


(* Question 6 *)

let e_out () =
  let learnset = create_dataset 100                in
  let f        = target ()                         in
  let labels   = get_labels learnset f             in
  let hypo     = solve learnset labels |> classify in
  let testset  = create_dataset 1000               in
  hypo_error (get_labels testset f) testset hypo


(* Question 7 *)

let perceptron_with_lr () =
  let learnset = create_dataset 10     in
  let f        = target ()             in
  let labels   = get_labels learnset f in
  let init_ws  = solve learnset labels in

  let find_mismatches err_vec =
    let ids = Vec.init (Vec.dim err_vec) float_of_int            in
    let mis = Vec.mul err_vec ids |> Vec.neg |> Vec.to_list      in
    (List.filter ((<) 0.) mis) |> List.map int_of_float          in

  let get_random_elem xs =
    List.nth xs (Random.int (List.length xs)) in

  let rec perceptron dataset labels ws iters =
    let hypo    = classify ws             in
    let labels' = get_labels dataset hypo in
    let err_vec = Vec.mul labels labels'  in (* 1 is match, -1 mismatch *)
    let err_ids = find_mismatches err_vec in (* indices of negative ones *)
    if err_ids == []
    then iters
    else
      let rand_miss = get_random_elem err_ids                      in
      let rand_pt   = Mat.copy_row dataset rand_miss               in
      let rand_lb   = labels.{rand_miss}                           in
      let delta_vec = Vec.map (( *. )  rand_lb) rand_pt            in
      perceptron dataset labels (Vec.add ws delta_vec) (iters+.1.) in

  perceptron learnset labels init_ws 0.


(* Question 8 *)

let square_classify ws pt =
  classify ws (Vec.map (fun x -> x *. x) pt)

let add_noise labels =
  let noise_to_int x = if x >= 0.8 then -1. else 1. in
  let noise_vec = Vec.random (Vec.dim labels) |> Vec.map noise_to_int in
  Vec.mul labels noise_vec

let e_in_nonlinear () =
  let learnset = create_dataset 1000                                  in
  let target   = square_classify (Vec.of_array [| (-0.6); 1.; 1. |])  in
  let labels   = get_labels learnset target |> add_noise              in
  let hypo     = solve learnset labels |> classify                    in
  hypo_error labels learnset hypo


(* Question 9 *)

let trans_cell orig row col =
  let x0 = orig.{row, 1} in
  let x1 = orig.{row, 2} in
  let x2 = orig.{row, 3} in
  match col with
    1 -> x0
  | 2 -> x1
  | 3 -> x2
  | 4 -> x1 *. x2
  | 5 -> x1**2.
  | 6 -> x2**2.
  | _ -> raise (Failure "Matrix has more than 6 cols")

let trans_dataset dataset =
  let m = Mat.dim1 dataset in
  let n = 6                in
  Mat.init_rows m n (fun r c -> trans_cell dataset r c)

let transformed =
  let learnset = create_dataset 1000                                 in
  let target   = square_classify (Vec.of_array [| (-0.6); 1.; 1. |]) in
  let labels   = get_labels learnset target |> add_noise             in
  let trans    = trans_dataset learnset                              in
  solve trans labels


(* Questions 10 *)

let trans_pt pt =
  let x0 = pt.{1} in
  let x1 = pt.{2} in
  let x2 = pt.{3} in
  Vec.of_array [| x0; x1; x2; x1 *. x2; x1**2.; x2**2. |]

let trans_hypo ws pt =
  classify ws (trans_pt pt)

let e_out_trans () =
  let testset = create_dataset 1000                                 in
  let target  = square_classify (Vec.of_array [| (-0.6); 1.; 1. |]) in
  let labels  = get_labels testset target |> add_noise              in
  let hypo    = trans_hypo transformed                              in
  hypo_error labels testset hypo


(* Print solutions *)

let _ = printf "\nQuestion 5:\n"
let _ = printf "avg error in the sample: \t%f\n"      (avg e_in)
let _ = printf "\nQuestion 6:\n"
let _ = printf "avg error out of the sample: \t%f\n"  (avg e_out)
let _ = printf "\nQuestion 7:\n"
let _ = printf "avg iterations of perceptron: \t%f\n" (avg perceptron_with_lr)
let _ = printf "\nQuestion 8:\n"
let _ = printf "avg e_in of nonlinear: \t\t%f\n"      (avg e_in_nonlinear)
let _ = printf "\nQuestion 9:\n"
let _ = printf "nonlinear weights vector: \n%a\n"     pp_vec transformed
let _ = printf "\nQuestion 10:\n"
let _ = printf "avg e_out of transformed: \t%f\n\n"   (avg e_out_trans)