%-Attribution de taches-

include "globals.mzn";

array[1..N,1..S] of int: cost;%cout d'une tache dependamment du sous-traitant
array[1..N,1..S] of int: duration;%duree d'une tache dependamment du sous-traitant, en jours

int: S;%nombre de sous-traitants
int: N;%nombre de tâches
int: T;%horizon temporel, en jours

% array[<A COMPLETER>] of var <A COMPLETER>: x;

% var int: C = <A COMPLETER>;

% <A COMPLETER>

output [
  "x: ", show(x), "\n",
  "Cost: ", show(C)]
;