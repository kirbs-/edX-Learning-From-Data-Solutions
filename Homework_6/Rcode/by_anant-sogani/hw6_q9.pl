#!/usr/bin/env perl

use strict;
use warnings;

# Number of units in Layer 0.
my $u0 = 10;

# Number of units in Layer L.
my $uL = 1;

# Total Budget of units in Hidden Layers.
my $B = 36;

# Maximum Hidden Layers possible.
my $MAXM = $B / 2;

# Hidden Layers in current Search Space.
my $M;

# Configurations Searched per M.
my @conf;

# Distribution of free units.
my @X;

# Distribution of total units: Y = X + 2.
my @Y;

# Configuration of Maximum Weights.
my ($maxw, @MAXW, $maxM);

# Configuration of Minimum Weights.
my ($minw, @MINW, $minM);

sub calc {

    $conf[$M]++;

    # Add the mandatory 2 units per Hidden Layer.
    $Y[$_] = (2 + $X[$_]) for (1 .. $M);

    # Calculate the number of weights.
    my $N = 0;

    my $L = $M + 1;
    for my $l (1 .. $L) {
        $N += ($Y[$l] - 1) * $Y[ $l - 1 ];
    }

    # Store configuration of Maximum Weights.
    if (!defined($maxw) || ($N > $maxw)) {
        ($maxw, $maxM, @MAXW) = ($N, $M, @Y);
    }

    # Store configuration of Minimum Weights.
    if (!defined($minw) || ($N < $minw)) {
        ($minw, $minM, @MINW) = ($N, $M, @Y);
    }
}

sub explore {

    # Layer, Budget.
    my ($l, $b) = @_;

    # Termination Condition - Last Hidden Layer.
    if ($l == $M) {
        $X[$l] = $b;
        calc();
        return;
    }

    # Explore.
    for my $b_ (0 .. $b) {
        $X[$l] = $b_;
        explore($l + 1, $b - $b_);
    }
}

print "Brute Force Search .... takes about 60 seconds\n";

# Explore all possible number of Hidden Layers.
for (1 .. $MAXM) {

    $M = $_;

    # Initialization.
    my $L = $M + 1;
    $Y[0]     = $u0;
    $Y[$L]    = $uL + 1;
    $conf[$M] = 0;

    # Exploration Budget (Minus the mandatory 2 units per Layer).
    my $b = $B - (2 * $M);

    # Search Space of 'M' Hidden Layers.
    explore(1, $b);
}

print "\nRESULTS FOR (d0 + 1) = $u0 and BUDGET = $B\n";
print_max();
print_min();

#print_details();

# Print Configuration of Maximum Weights.
sub print_max {
    print '-' x 50 . "\n";
    print "Maximum Weights = $maxw for M = $maxM Hidden Layers:\n";
    print $MAXW[$_] . " " for (1 .. $maxM);
}

# Print Configuration of Minimum Weights.
sub print_min {
    print "\n" . '-' x 50 . "\n";
    print "Minimum Weights = $minw for M = $minM Hidden Layers:\n";
    print $MINW[$_] . " " for (1 .. $minM);
    print "\n";
}

# Search Details.
sub print_details {
    print "\n" . '-' x 50 . "\n";
    print "Configurations Searched:\n";
    my $sum = 0;
    for my $i (1 .. $MAXM) {
        $sum += $conf[$i];
        printf("M = %2d : searches = %7d, total = %7d\n", $i, $conf[$i], $sum);
    }
}
