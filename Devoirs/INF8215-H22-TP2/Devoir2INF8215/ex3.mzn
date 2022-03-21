%-Attribution de taches-

%Utiliser solver COIN-BC
include "globals.mzn";

array[1..N,1..S] of int: cost;%cout d'une tache dependamment du sous-traitant
array[1..N,1..S] of int: duration;%duree d'une tache dependamment du sous-traitant, en jours

int: S;%nombre de sous-traitants
int: N;%nombre de tâches
int: T;%horizon temporel, en jours

%Tableau dont l'indice est une tache et la valeur est le sous-traitant assossié à la tache
array[1..N] of var 1..S: x;

%Sous-traitant tache 1 et tache 2 ne peuvent s'occupper d'aucune autre tache
constraint x[1] != x[2];
constraint forall(t in 3..N)(x[t] != x[1] /\ x[t] != x[2]);

% attribution des tâches doit respecter l’horizon temporel fixé T
constraint forall(s in 1..S)(sum(t in 1..N)(if x[t] == s then duration[t,s] else 0 endif) <= T);

%Somme des coûts
var int: C = sum(n in 1..N)(cost[n, x[n]]);

solve minimize C;

output [
  "x: ", show(x), "\n",
  "Cost: ", show(C),]
;
% RUN en utilisant le solveur COIN-BC
% Résultat Cout total minimum permettant de finir dans les temps est de 2304.