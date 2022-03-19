%-Enigme des mariages-

include "globals.mzn"; 

int: n = 5;

int: Dupont = 1;
int: Tremblay = 2;
int: Durand = 3;
int: Gagnon = 4;
int: Petit = 5;

var 1..n: Remy;
var 1..n: David;
var 1..n: Charles;
var 1..n: Pierre;
var 1..n: Edouard;
array[1..n] of var 1..n: M = [Remy, David, Charles, Pierre, Edouard];
 
var 1..n: Emilie;
var 1..n: Juliette;
var 1..n: Maria;
var 1..n: Elisabeth;
var 1..n: Manon;
array[1..n] of var 1..n: F = [Emilie, Juliette, Maria, Elisabeth, Manon];

array[1..n] of var 0..100: married;

% <A COMPLETER>

output [
  "M: ", show(M), "\n",
  "F: ", show(F), "\n",
  "married : ", show(married), "\n",
]
;