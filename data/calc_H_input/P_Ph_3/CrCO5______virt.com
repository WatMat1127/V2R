# MIN/wB97XD/Def2SVP

0 1
Cr                 0.00000000    0.00000000   -2.18000000
C                  1.90000000    0.00000000   -2.18000000
C                  0.00000000    1.90000000   -2.18000000
C                 -1.90000000    0.00000000   -2.18000000
C                  0.00000000   -1.90000000   -2.18000000
C                  0.00000000    0.00000000   -4.08000000
O                  3.01540000    0.00000000   -2.18000000
O                  0.00000000    3.01540000   -2.18000000
O                 -3.01540000    0.00000000   -2.18000000
O                  0.00000000   -3.01540000   -2.18000000
O                  0.00000000    0.00000000   -5.19540000
P                  0.00000000    0.00000000    0.00000000
Cl                 1.76678313   -0.76003524    0.68001749
Cl                -0.22692110    1.90880982    0.68304603
Cl                -1.54135703   -1.15220715    0.67693218
Options
GauProc=16
GauMem=800
EigenCheck
MinFreqValue=50.0
@@SubAddExPot@@
MaxOptITR=999

