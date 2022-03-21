%-Carres-

include "globals.mzn";

array[1..5] of var 10..100: x;
% constraint <A COMPLETER>;

constraint sum(i in 1..4) (x[i]^2) = x[5]^2;
constraint alldifferent(x);
% solve <A COMPLETER>;

solve minimize x[5]^2;

output [
  show(x)
];
% Les 4 entiers sont 17, 14, 12, et 10 tel que 17^2 + 14^2 + 12^2 + 10^2 = 27^2