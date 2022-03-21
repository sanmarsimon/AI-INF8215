%-2p Quoridor tournament-

include "globals.mzn";

int: MAX_T = 30;%nombre maximum de creneaux
int: G = 3;%nombre de tables de tournoi
int: P = 13;%nombre de joueurs

var 1..MAX_T: T;%temps a minimiser

array[1..MAX_T,1..G,1..2] of var 0..P: x;

array[1..P,1..P] of var 0..MAX_T: duels;


% <A COMPLETER>
%Un joueur ne peut pas jouer contre lui meme
constraint forall(j in 1..P)(duels[j,j] = 0);

%Un joueur ne peut pas jouer sur deux craineaux en meme temps
constraint forall(j in 1..P)(alldifferent([duels[j,a] | a in 1..P]) /\ alldifferent([duels[a,j] | a in 1..P])); 

%Les joueurs ne jouent pas trois créneaux de suite
%On scan la matrice de duels par ligne, pour chaque valeur dans la ligne (creneau) on vérifie qu'il n'y a pas en meme temps le creneau suivant et
%celui d'apres
constraint forall(i,j in 1..P)(not( duels[i,j]+1 in row(duels,i) /\ duels[i,j]+2 in row(duels,i)));

T=max(array1d(duels));

%Maximum 2*G memes creneaux pour respecter le nombre de table maximal par creneau (2 joueurs par table)
constraint forall(c in 1..T)(count(array1d(duels), c) <= G);


solve minimize T;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% On construit le tableau de match pour chaque creneau et chaque table (on attribue des tables)

constraint forall(j in 2..P)(x[duels[1,j],1,1] = 1);
constraint forall(j in 2..P)(x[duels[1,j],1,2] = j);


% if x[duels[i,j],1,1] != 0 then x[duels[i,j],1,1] = i elseif x[duels[i,j],2,1] != 0 then x[duels[i,j],2,1] = i else x[duels[i,j],3,1] = i endif
% );
% constraint forall(i,j in 2..P)(
% if x[duels[i,j],1,2] != 0 then x[duels[i,j],1,2] = j elseif x[duels[i,j],2,2] != 0 then x[duels[i,j],2,2] = j else x[duels[i,j],3,2] = j endif
% );

output

["Matrice de match \n"]
++
["Player \t |  "]
++
[show(i)++ "\t" | i in 1..P ]
++
["-------------------------------------------------------------------------------------------------------------------------- \n"]
++
[
if j == 1 then "\n" ++ show(i)++"\t |  " endif ++ show(duels[i,j]) ++ "\t" | i,j in 1..P
]
++
["-------------------------------------------------------------------------------------------------------------------------- \n"]
++
["Schedule ("++show(T)++" time slots)\n"]
++
[
    if j = 1 then "\n" else " " endif ++ show(x[i,j,1])++" "++show(x[i,j,2])++ if j = G then " " else " -" endif
      | i in 1..MAX_T, j in 1..G
]