%-2p Quoridor tournament-

include "globals.mzn";

int: MAX_T = 30;%nombre maximum de creneaux
int: G = 3;%nombre de tables de tournoi
int: P = 13;%nombre de joueurs

var 1..MAX_T: T;%temps a minimiser

array[1..MAX_T,1..G,1..2] of var 0..P: x;


%Il s'agit d'un tableau de deux dimensions presentant une matrice de match. Ainsi, chaque case (i,j) dans le tableau correspond au creneau du match entre le joueur i et le joueur j quand i est different de j. Il y a un maximum de G memes creneauxm, ansi, cette matrice presente le calendrier complet des matchs avec leur creneau. La table importe peu pour chacun des joueurs, sauf pour le joueur 1 dont la table est la numero 1.
array[1..P,1..P] of var 0..MAX_T: duels;


% <A COMPLETER>
%Un joueur ne peut pas jouer contre lui meme
constraint forall(j in 1..P)(duels[j,j] = 0);

%Le joueur 1 joue les matchs en ordre
constraint forall(j in 1..P-1)(duels[1,j] < duels[1,j+1]);
constraint duels[1,2] = 1;
constraint duels[1,3] = 2;
% Les matchs mirroirs sont sur le meme creneau
constraint forall(i,j in 1..P)(duels[i,j] = duels[j,i]);

%Un joueur ne peut pas jouer sur deux craineaux en meme temps
constraint forall(j in 1..P)(alldifferent([duels[j,a] | a in 1..P]) /\ alldifferent([duels[a,j] | a in 1..P])); 

%Les joueurs ne jouent pas trois créneaux de suite
%On scan la matrice de duels par ligne, pour chaque valeur dans la ligne (creneau) on vérifie qu'il n'y a pas en meme temps le creneau suivant et celui d'apres
constraint forall(i,j in 1..P)(if i != j then (not(duels[i,j]+1 in row(duels,i) /\ duels[i,j]+2 in row(duels,i))) endif);

%Maximum 2*G memes creneaux pour respecter le nombre de table maximal par creneau (2 joueurs par table)
constraint forall(c in 1..T)(count(array1d(duels), c) <= 2*G);

%On veut minimiser le numero maximal de creneau dans la matrice de match
T=max(array1d(duels));
solve minimize T;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% On construit le tableau de match pour chaque creneau et chaque table (on attribue des tables)

%On force le joueur 1 à la table 1 
constraint forall(j in 2..P)(x[duels[1,j],1,1] = 1);
constraint forall(j in 2..P)(x[duels[1,j],1,2] = j);


%On attribue aux autres duels une table libre dans le creneau
%Il faut que la premiere occurence de chaque creneau soit à la table 1, ensuite la deuxième à la table 2, ensuite la troisième à la table 3. On n'a pas réussi à trouver comment modéliser ceci sur minizinc.

%constraint forall(i,j in 2..P)()
% constraint forall(i in 1..P-1, j in i+1..P)(%On loop sur les elements sub-diagonaux (pour eviter les doublons)
%   x[duels[i,j],1,1] = i
% );


output

["Matrice de match \n"]
++
["Player \t |  "]
++
[show(i)++ "\t" | i in 1..P] ++ ["\n \t"]
++
["---\t" | i in 1..P] ++ ["\n"]
++
[
if j == 1 then "\n" ++ show(i)++"\t |  " endif ++ show(duels[i,j]) ++ "\t" | i,j in 1..P
]
++
["\n Schedule ("++show(T)++" time slots)\n"]
++
[
    if j = 1 then "\n" else " " endif ++ show(x[i,j,1])++" "++show(x[i,j,2])++ if j = G then " " else " -" endif
      | i in 1..MAX_T, j in 1..G
]