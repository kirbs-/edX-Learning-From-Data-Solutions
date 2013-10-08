<?php

// written by Henry Feldman    henry@conceptual.com
// for Caltech Machine Learning Class

function sign( $number ) { 
    return ( $number > 0 ) ? 1 : ( ( $number < 0 ) ? -1 : 0 ); 
} 

function getrand()
{
	$r = rand(-10000000,10000000) / 10000000;
	return ($r);
}

// this is the 'work' function.  pass it # of points.   It returns # of iterations
//      also gives back the "error_rate" done by running lots of points

function do_a_run($count,&$error_rate)
{

	// generate the +1 / -1 dividing line from 2 random points
	
	$p1x = getrand();
	$p1y = getrand();
	$p2x = getrand();
	$p2y = getrand();

	$slope = ($p2y - $p1y) / ($p2x - $p1x);
	$intercept = $p1y - $slope * $p1x;

	// initial the weight array
	
	$w = array(0,0,0);

	// build the sample data point array
	$data = array();

	for ($i=0; $i < $count; $i++) {
		$data[$i] = array();
	
		// element 0 is always 1
		$data[$i][0] = 1;
	
		$x1 = getrand();
		$real_y = getrand();
		$calc_y = $slope * $x1 + $intercept;
		if ($real_y > $calc_y)
			$result = 1;
		else
			$result = -1;
			
		// stuff the values of the X1, X2 (real_y) and the +1/-1 result
		$data[$i][1] = $x1;
		$data[$i][2] = $real_y;
		$data[$i][3] = $result;
	
	}

	// At first, I was too lazy to add code to pick random mislocated points.
	// just started at element 0 and and looked for the first missed point
	for ($loop=0; $loop < 30000; $loop++) {
		for ($i=0; $i < $count; $i++) {
			$real_result = $data[$i][3];
	
			// calculate our predicted result
			$calc_result = 0;
			for ($j=0; $j <= 2; $j++)
				$calc_result += $w[$j] * $data[$i][$j];
				
			// if we didn't match sign, adjust weight and start over
			if (sign($calc_result) != $real_result) {
				for ($j=0; $j <= 2; $j++)
					$w[$j] += $real_result * $data[$i][$j];
				break;
			}
		}
		
		// if we got all the way through without finding a mismatch, then we are done
		if ($i == $count)
			break;
	}
	
	//echo "iterations to converge  $loop\n";

	// TRY RANDOM POINTS.   SEE how we agree
	$test_count = 10000;
	$error_count = 0;
	for ($num = 0; $num < $test_count; $num++) {

		// new random point
		$x1 = getrand();
		$real_y = getrand();
		$calc_y = $slope * $x1 + $intercept;
		if ($real_y > $calc_y)
			$result = 1;
		else
			$result = -1;
			
		$calc_result = $w[0] + $x1 * $w[1] + $real_y * $w[2];
		$sign_calc_result = sign($calc_result);
		if ($sign_calc_result != $result) {
			$error_count++;
		}
	}

	$error_rate = $error_count / $test_count;
	//echo "ERROR RATE   $error_rate\n";
	return ($loop);
}

// find average number of iterations to converge
$loop_total = 0;
$error_total = 0;
$loop_tries = 1000;
$point_count = 100;

for ($loopnum = 0; $loopnum < $loop_tries; $loopnum++) {
	$loops = do_a_run($point_count,$error_rate);
	$loop_total += $loops;
	$error_total += $error_rate;
}

$loop_average = $loop_total / $loop_tries;
$error_average = $error_total / $loop_tries;
echo "Average Number of Loops  $loop_average\n";
echo "Average Errors  $error_average\n";


?>

