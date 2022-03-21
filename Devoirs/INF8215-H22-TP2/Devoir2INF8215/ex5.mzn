%-4p Quoridor tournament-

include "globals.mzn";

int: T = 14;%jours de tournoi
int: G = 2;%sites de jeu
int: P = 13;%nombre de joueurs

array[1..T,1..G,1..4] of var 1..P: x;

% <A COMPLETER>
array[1..P] of var 0..11: matchsParJoueur;
array[1..P,1..T,1..4] of var 0..P: joueursRencontres;
array[1..P,1..T] of var 1..G: siteVisite;
array[1..P] of var 1..22: sommeSiteVisite;
%constraint forall(p in 1..P)(sum(t in 1..T,g in 1..G,j in 1..4)(if x[t,g,j] == p then 1 endif));

% 4 Joueuers différents par partie
constraint forall(t in 1..T,g in 1..G)(x[t,g,1] != x[t,g,2] /\ x[t,g,1] != x[t,g,3] /\ x[t,g,1] != x[t,g,4] /\ x[t,g,2] != x[t,g,3] /\ x[t,g,2] != x[t,g,4] /\ x[t,g,3] != x[t,g,4]);
% Un joueur ne peut être que sur un site par jour
constraint forall(t in 1..T,j in 1..4)(x[t,1,j] != x[t,2,1] /\ x[t,1,j] != x[t,2,2] /\ x[t,1,j] != x[t,2,3] /\ x[t,1,j] != x[t,2,4]);
% 1 jour de repos entre deux matchs sur deux sites
constraint forall(t in 1..T-1,j in 1..4)(x[t,1,j] != x[t+1,2,1] /\ x[t,1,j] != x[t+1,2,2] /\ x[t,1,j] != x[t+1,2,3] /\ x[t,1,j] != x[t+1,2,4]);
constraint forall(t in 1..T-1,j in 1..4)(x[t,2,j] != x[t+1,1,1] /\ x[t,2,j] != x[t+1,1,2] /\ x[t,2,j] != x[t+1,1,3] /\ x[t,2,j] != x[t+1,1,4]);
% Chaque joueur joue entre 8 et 11 matchs
constraint forall(p in 1..P)(matchsParJoueur[p] = count(x,p));
constraint forall(p in 1..P)(matchsParJoueur[p] >= 8 /\ matchsParJoueur[p] <= 11);

% Chaque joueur a joué au moins une fois sur chacun des deux sites
%constraint forall(t in 1..T,p in 1..P,g in 1..2,j in 1..4)(if x[t,g,j] == p then siteVisite[p,t] = g endif);
%constraint forall(p in 1..P)(sommeSiteVisite[p] = sum(t in 1..T)(siteVisite[p,t]));
%constraint forall(p in 1..P)(sommeSiteVisite[p] != matchsParJoueur[p] /\ sommeSiteVisite[p] != matchsParJoueur[p]*2);

constraint forall(t in 1..T,p in 1..P)(joueursRencontres[p,t,1] = p);
constraint forall(p in 1..P,t in 1..T,g in 1..G,j in 1..4)(if x[t,g,j] == p then joueursRencontres[p,t,1] = p endif);

solve satisfy;

output
["\n Site 1 \t \t Site 2\n"]
++
[
    "\n"++show(x[i,1,1])++" "++show(x[i,1,2])++" "++show(x[i,1,3])++" "++show(x[i,1,4])++" \t \t"++show(x[i,2,1])++" "++show(x[i,2,2])++" "++show(x[i,2,3])++" "++show(x[i,2,4])
  | i in 1..T
]
% RUN USING COIN-BC
% RESULT :
% Site 1 	 	     Site 2

% 3 11 1 8 	 	  13 5 9 2
% 12 6 11 10 	 	9 7 13 4
% 5 8 2 6 	 	   9 3 1 4
% 5 12 10 2 	 	 7 11 3 1
% 13 10 12 6 	 9 11 7 8
% 6 10 12 2 	  	9 8 5 3
% 10 4 1 2 	 	  13 5 7 3
% 9 1 4 11 	   3 6 13 8
% 9 7 12 11 	 	 8 10 3 2
% 1 12 4 6 	 	  5 13 8 2
% 9 4 12 10 	 	 5 11 13 8
% 7 9 4 2 	 	   6 1 3 11
% 10 12 7 2 	 	 1 3 13 5
% 4 7 6 2 	 	   3 1 8 5