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
constraint alldifferent(married);
constraint alldifferent(M);
constraint alldifferent(F);


% Juliette n’est pas mariée depuis aussi longtemps que Charles ou les Tremblays, mais depuis plus longtemps que David ou les Duponts
constraint married[Juliette] < married[Charles]; %ok
constraint married[Juliette] < married[Tremblay]; %ok
constraint married[Juliette] > married[David];%ok
constraint married[Juliette] > married[Dupont]; %ok

% Élisabeth est mariée depuis deux fois plus longtemps que les Durands, mais deux fois moins longtemps que Rémy
constraint married[Elisabeth] = 2*married[Durand]; %ok
constraint married[Elisabeth] = married[Remy]/2; %ok

% Les Petits sont mariés depuis 10 ans de plus que Pierre et 10 ans de moins que Maria.
constraint married[5] = married[Pierre] + 10; %ok
constraint married[5] = married[Maria] - 10; %ok

% David et Manon sont mariés depuis 25 ans de moins que les Gagnons qui, mariés depuis 30 ans, sont le couple marié depuis le plus longtemps.
constraint David = Manon; 
constraint married[Gagnon] = 30; %ok
constraint married[David] = married[Gagnon] - 25; %ok

constraint arg_max(married) = Gagnon;

% Ni Émilie ni les Duponts ont été mariés depuis le moins longtemps.
constraint arg_min(married) != Emilie;
constraint arg_min(married) != Dupont;

% Édouard est marié depuis 25 ans.
constraint married[Edouard] = 25;

solve satisfy;
output [
  "M: ", show(M), "\n",
  "F: ", show(F), "\n",
  "married : ", show(married), "\n",
]
;