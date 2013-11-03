/*
 author: __________
 date: 10/10/2013
 Goal of this Program: 
 Simulate 10 tosses of each of 1000 fair coins and find fraction of tosses 
 that come up heads for the first coin, a randomly selected coin, and the coin with the fewest heads.
 Perform the experiment 100,000 times and average the results.
 Assumptions: 
 rand pseudorandomly generates each bit independently. 
 Each bit that rand might set to 1 is 1 in RAND_MAX.
 */
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <string.h>     /* memset */
#include <math.h>       /* pow */

#define NUM_COINS 1000

#define NUM_TOSSES 10

#define NUM_EXPERIMENTS 100000

double compute_expected_fraction_heads_min();
double compute_combinations(int choices, int chosen);

int main ()
{
	int head_count_array[NUM_COINS];
	unsigned int experiment_index;
	unsigned int coin_index;
	unsigned int toss_index;
	int bitshift = 0;
	int bitmask = 1;
	int current_coins;
	int coin_index_first;
	int coin_index_random;
	int coin_index_min;
	long total_head_count_first;
	long total_head_count_random;
	long total_head_count_min;
	long total_random_index;
	long num_toss_experiments;
	double fraction_heads_first;
	double fraction_heads_random;
	double fraction_heads_min;
	double average_random_index;
	
	/* initialize random seed: */
	srand (time(NULL));
	
	coin_index_first = 0;//in every experiment
	total_head_count_first = 0;
	total_head_count_random = 0;
	total_head_count_min = 0;
	total_random_index = 0;
	
	experiment_index = 0;
	while (experiment_index < NUM_EXPERIMENTS) {
		
		
		//set the head counts to zero at the start of the experiment:
		memset( head_count_array, 0, sizeof head_count_array );
		coin_index_random = rand()%NUM_COINS;//(This is not uniformly distributed.) ( NUM_COINS*rand() )/RAND_MAX;(Overruns allowed number of bits.)
		coin_index_min = 0;
		
		coin_index = 0;
		while (coin_index < NUM_COINS) {
			
			
			toss_index = 0;
			while (toss_index < NUM_TOSSES) {
				
				//If all of the bits that are 1 in RAND_MAX get shifted out, it is time to
				if (!(RAND_MAX >> bitshift)) {
					current_coins = rand();//start on a new random number
					bitshift = 0;//and start with its least significant bit.
				}
				
				//Add a heads to the total if (bitshift+1)th bit is 1:
				head_count_array[coin_index] += (current_coins >> bitshift)&bitmask;
				bitshift++;
				
				toss_index++;
			}
			
			//If the new total is left, make it the min. In the case of a tie, keep the first.
			if (head_count_array[coin_index] < head_count_array[coin_index_min]) {
				coin_index_min = coin_index;
			}
			
			coin_index++;
		}
		
		total_head_count_first += (long)head_count_array[coin_index_first];
		total_head_count_random += (long)head_count_array[coin_index_random];
		total_head_count_min += (long)head_count_array[coin_index_min];
		total_random_index += (long)coin_index_random;
		
		
		experiment_index++;
	}
	
	
	
	num_toss_experiments = NUM_EXPERIMENTS*NUM_TOSSES;
	fraction_heads_first = (double)total_head_count_first/num_toss_experiments;
	fraction_heads_random = (double)total_head_count_random/num_toss_experiments;
	fraction_heads_min = (double)total_head_count_min/num_toss_experiments;
	average_random_index = (double)total_random_index/NUM_EXPERIMENTS;
	
	printf( "%d experiments complete, %d coins tossed per experiment, each coin tossed %d times, average fraction heads of first coin = %f, of randomly selected coin (average index %f) = %f, and of min coin with fewest head in current experiment = %f (expected value = %f)\n", NUM_EXPERIMENTS, NUM_COINS, NUM_TOSSES, fraction_heads_first, average_random_index, fraction_heads_random, fraction_heads_min, compute_expected_fraction_heads_min() );
	
	return 0;
}

/*
 Because, on a single coin toss on a fair coin, heads and tails are equally likely, 
 P(NUM_TOSSES coin tosses produce a particular sequence, such as, for NUM_TOSSES = 10, HTTHHHTTHH.) = (1/2)^NUM_TOSSES.
 Each specific sequence is a mutually exclusive possible outcome, and they all have the same probability, so we can find the probability of getting a certain number of heads by multiplying that probability by the number of sequences with that number of heads.
 P(Out of NUM_TOSSES coin tosses, num_heads tosses come up heads.) = (number of NUM_TOSSES long sequences of heads and tails that have num_heads tails)/(number of distinct NUM_TOSSES long sequences of heads and tails) = ( NUM_TOSSES!/(num_heads! * (NUM_TOSSES-num_heads)!) )*(1/2)^NUM_TOSSES.
 Because getting a certain number of heads in NUM_TOSSES tosses and getting a different number of heads in NUM_TOSSES tosses are mutually exclusive possibilities, we can find the probability that the number of heads will be in a certain range by adding the probabilities of the numbers in that range.
 P(Out of NUM_TOSSES coin tosses, more than num_heads tosses come up heads.) = Sum over greater_num_heads=num_heads+1 to NUM_TOSSES of P(Out of NUM_TOSSES coin tosses, greater_num_heads tosses come up heads).
 If we now take NUM_COINS coins and toss them NUM_TOSSES times each, because each set of NUM_TOSSES tosses is an independent event, the probability of getting a specific sequence of outcomes, such as, for NUM_COINS = 5, coin 1 gets 5 heads, coin 2 gets more than 5 heads, coin 3 gets more than 5 heads, coin 4 gets 5 heads, and coin 5 gets 5 heads 
 is equal to the product of the probabilities of the individual outcomes, in this case P(a coin gets 5 heads)*P(a coin gets >5 heads)*P(a coin gets >5 heads)*P(a coin gets 5 heads)*P(a coin gets 5 heads).
 Because getting num_heads heads and getting more than num_heads heads are mutually exclusive, each sequence of outcomes in which equal_coins coins get num_heads heads and all other coins get more than num_heads heads represents an equally probable, mutually exclusive outcome. Thus,
 P(Out of NUM_COINS coins, each tossed NUM_TOSSES times, equal_coins coins come up heads num_heads times and NUM_COINS-equal_coins coins come up heads more than num_heads times.) = (number of ways to choose equal_coins coins out of NUM_COINS coins)*(P(A coin comes up heads num_heads times out of NUM_TOSSES tries.)^equal_coins)*(P(A coin comes up heads more than num_heads times out of NUM_TOSSES tries.)^(NUM_COINS - equal_coins)).
 Because getting a certain number of coins that come up heads num_heads times with the rest coming up heads more times than that and getting a different number of coins that come up heads num_heads times with all others coming up heads more than that are mutually exclusive possibilities,
 P(Out of NUM_COINS coins, each tossed NUM_TOSSES times, one or more comes up heads num_heads times and all others come up heads more than num_heads times.) = Sum over equal_coins=1 to NUM_COINS of P(Out of NUM_COINS coins, each tossed NUM_TOSSES times, equal_coins coins come up heads num_heads times, and all others come up heads more than num_heads times).
 For a particular num_heads to be the minimum, it must occur for one or more coins, and all other coins must get more than num_heads heads.
 Thus, Expected_Value(num_heads) = Sum over num_heads=0 to 10 of num_heads*P(Out of NUM_COINS coins, each tossed NUM_TOSSES times, one or more comes up heads num_heads times and all others come up heads more than num_heads times.).
 To get the heads fraction, we take num_heads/NUM_TOSSES.
 */
double compute_expected_fraction_heads_min() {
	int num_heads;
	int greater_num_heads;
	int equal_coins;
	int factorials[NUM_TOSSES+1];
	double probability_of_sequence;//of specific coins coming up heads or tails
	double num_heads_probabilities[NUM_TOSSES+1];
	double greater_than_num_heads_probabilities[NUM_TOSSES+1];
	double ways_to_pick_equal_coins_coins[NUM_COINS+1];
	double num_heads_is_min_probabilities[NUM_TOSSES+1];
	double expected_value;
	
	num_heads = 0;
	//printf("num_heads:\t");
	while (num_heads <= NUM_TOSSES) {
		//printf("%d\t", num_heads);
		num_heads++;
	}
	//printf("\n");
	
	factorials[0] = 1;
	num_heads = 1;
	//printf("num_heads!:\t");
	while (num_heads <= NUM_TOSSES) {
		factorials[num_heads] = num_heads*factorials[num_heads - 1];
		//printf("%d\t", factorials[num_heads]);
		num_heads++;
	}
	//printf("\n");
	
	probability_of_sequence = 1.0/pow(2.0, NUM_TOSSES);
	//printf("probability of a specific sequence of heads and tails = %f\n", probability_of_sequence);
	
	num_heads = 0;
	//printf("P(num_heads=):\t");
	while (num_heads <= NUM_TOSSES) {
		num_heads_probabilities[num_heads] = probability_of_sequence*factorials[NUM_TOSSES]/( factorials[num_heads] * factorials[NUM_TOSSES - num_heads] );
		//printf("%f\t", num_heads_probabilities[num_heads]);
		num_heads++;
	}
	//printf("\n");
	
	num_heads = 0;
	//printf("P(num_heads>):\t");
	while (num_heads <= NUM_TOSSES) {
		greater_than_num_heads_probabilities[num_heads] = 0;
		greater_num_heads = num_heads+1;
		while (greater_num_heads <= NUM_TOSSES) {
			greater_than_num_heads_probabilities[num_heads] += num_heads_probabilities[greater_num_heads];
			greater_num_heads++;
		}
		//printf("%f\t", greater_than_num_heads_probabilities[num_heads]);
		num_heads++;
	}
	//printf("\n");
	
	equal_coins = 1;
	//printf("equal_coins:\t");
	while (equal_coins <= NUM_COINS) {
		//printf("%d\t", equal_coins);
		equal_coins++;
	}
	//printf("\n");
	
	equal_coins = 1;
	//printf("ways to pick equal_coins coins:\t");
	while (equal_coins <= NUM_COINS) {
		ways_to_pick_equal_coins_coins[equal_coins] = compute_combinations(NUM_COINS, equal_coins);
		//printf("%d:%f\t", equal_coins, ways_to_pick_equal_coins_coins[equal_coins]);
		equal_coins++;
	}
	//printf("\n");
	
	//printf("P(min heads is num_heads):\t");
	num_heads = 0;
	while (num_heads <= NUM_TOSSES) {
		equal_coins = 1;
		num_heads_is_min_probabilities[num_heads] = 0;
		while (equal_coins <= NUM_COINS) {
			num_heads_is_min_probabilities[num_heads] += ways_to_pick_equal_coins_coins[equal_coins]*pow(num_heads_probabilities[num_heads], equal_coins)*pow(greater_than_num_heads_probabilities[num_heads], NUM_COINS-equal_coins);
			equal_coins++;
		}
		//printf("%f\t", num_heads_is_min_probabilities[num_heads]);
		num_heads++;
	}
	//printf("\n");
	
	num_heads = 0;
	expected_value = 0;
	while (num_heads <= NUM_TOSSES) {
		expected_value += num_heads*num_heads_is_min_probabilities[num_heads];
		num_heads++;
	}
	//printf("Expected Value of num_heads=%f\n", expected_value);
	
	return expected_value/NUM_TOSSES;
}

//This will find choicesCchosen +/- some rounding error.
double compute_combinations(int choices, int chosen) {
	if (chosen == 0) {
		return 1;
	}
	else {
		return choices*compute_combinations(choices-1, chosen-1)/chosen;
	}
}