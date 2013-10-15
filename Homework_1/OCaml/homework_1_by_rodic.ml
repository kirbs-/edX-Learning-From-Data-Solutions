open Printf
open Random

type point  = { x : float; y : float }
type vec3d  = { v0 : float; v1 : float; v2 : float }

let random_between x y () =
  Random.self_init (); (min x y) +. (Random.float (abs_float(x-.y)))

let get_random_coord = random_between (-1.) 1.

let get_point () = 
  { x = get_random_coord (); y = get_random_coord () }

let rec get_points n =
  if n <= 0
  then []
  else get_point () :: get_points (n-1)

let get_target =
  let p1 = get_point ()
  and p2 = get_point () in
  let m  = (p1.y -. p2.y) /. (p1.x -. p2.x) in
  let b  = p1.y -. m *. p1.x in
  { v0 = b; v1 = m; v2 = (-1.0) }

let sign pt ws =
  if ws.v0 +. ws.v1 *. pt.x +. ws.v2 *. pt.y < 0.0
  then (-1.)
  else 1.

let label_pts ps ws =
  let labels = Hashtbl.create 100 in
  let rec aux = function
      []    -> labels
    | p::ps -> Hashtbl.add labels p (sign p ws); aux ps in
  aux ps

let is_misclassified ws labels pt =
  Hashtbl.find labels pt <> sign pt ws

let find_misclassified hypo labels pts =
  List.filter (is_misclassified hypo labels) pts

let pick_random_elem xs =
  List.nth xs (Random.int (List.length xs))

let update_hypo hypo labels pt =
  let klas = Hashtbl.find labels pt in
  let dh   = { v0 = 1. *. klas; v1 = pt.x *. klas; v2 = pt.y *. klas } in
  { v0 = hypo.v0+.dh.v0; v1 = hypo.v1+.dh.v1; v2 = hypo.v2+.dh.v2 } 

let perceptron ps target =
  let labels = label_pts ps target in
  let hypo   = { v0 = 0.; v1 = 0.; v2 = 0. } in 
  let rec aux ps hypo iters =
    match find_misclassified hypo labels ps with
      [] -> (hypo, iters)
    | ms ->
      let rand_pt = pick_random_elem ms in
      aux ps (update_hypo hypo labels rand_pt) (iters +. 1.)
  in aux ps hypo 0.

let e_out tws hws =
  let pts_num = 100. in
  let rec aux = function
      0. -> 0.
    | n  ->
      let p = get_point () in
      if sign p tws <> sign p hws
      then 1. +. aux (n-.1.)
      else aux (n-.1.) in
  aux pts_num /. pts_num

let main () =
  let pts_num  = 300
  and runs     = 1000. in
  let rec aux it_acc er_acc = function
      0.  -> (it_acc /. runs, er_acc /. runs)
    | run ->
      let target = get_target in
      let (h, i) = perceptron (get_points pts_num) target in
      aux (it_acc +. i) (er_acc +. e_out target h) (run-.1.) in
  let (avg_i, avg_e) = aux 1. 0. runs in
  printf "avg num of iters for %d points: %f\n" pts_num avg_i; 
  printf "avg error out of the sample: %f\n" avg_e; ()

let _ = main ()