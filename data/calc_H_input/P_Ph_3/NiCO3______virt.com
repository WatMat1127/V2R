# MIN/wB97XD/Def2SVP

0 1
Ni                -0.00000000   -0.00000000   -2.23386274
C                  0.87532582   -1.48571971   -2.74442791
C                  0.84928530    1.50027025   -2.74538086
C                 -1.72401276   -0.01528074   -2.74534277
O                  1.43218899   -2.43078784   -3.06226140
O                  1.38977657    2.45458614   -3.06384836
O                 -2.82073217   -0.02522892   -3.06376393
P                  0.00000000    0.00000000    0.00000000
Cl                -0.95042073   -1.67208751    0.68001749
Cl                 1.92219362    0.01482605    0.68304603
Cl                -0.97501946    1.65912624    0.67693218
Options
GauProc=16
GauMem=800
EigenCheck
MinFreqValue=50.0
@@SubAddExPot@@
MaxOptITR=999

