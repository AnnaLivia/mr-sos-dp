\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  - 3.5800429099300064e+07 Y0_0_1_1_0 - 3.5800429099300064e+07 Y0_0_1_1_1
   - 3.5800429099300064e+07 Y0_0_1_1_2 - 3.3487308569600027e+07 Y0_0_1_2_0
   - 3.3487308569600027e+07 Y0_0_1_2_1 - 3.3487308569600027e+07 Y0_0_1_2_2
   - 5.9385819794200011e+07 Y0_0_2_1_0 - 5.9385819794200011e+07 Y0_0_2_1_1
   - 5.9385819794200011e+07 Y0_0_2_1_2 - 5.8954186635099977e+07 Y0_0_2_2_0
   - 5.8954186635099977e+07 Y0_0_2_2_1 - 5.8954186635099977e+07 Y0_0_2_2_2
   - 3.1588684444799982e+07 Y0_1_1_2_0 - 3.1588684444799982e+07 Y0_1_1_2_1
   - 3.1588684444799982e+07 Y0_1_1_2_2 - 5.0545409460799985e+07 Y0_1_2_2_0
   - 5.0545409460799985e+07 Y0_1_2_2_1 - 5.0545409460799985e+07 Y0_1_2_2_2
   - 2.9796345709400035e+07 Y1_0_2_1_0 - 2.9796345709400035e+07 Y1_0_2_1_1
   - 2.9796345709400035e+07 Y1_0_2_1_2 - 2.9601317300500013e+07 Y1_0_2_2_0
   - 2.9601317300500013e+07 Y1_0_2_2_1 - 2.9601317300500013e+07 Y1_0_2_2_2
   - 2.8326389580599979e+07 Y1_1_2_2_0 - 2.8326389580599979e+07 Y1_1_2_2_1
   - 2.8326389580599979e+07 Y1_1_2_2_2
Subject To
 R0: X0_0_0 + X0_0_1 + X0_0_2 = 1
 R1: X0_1_0 + X0_1_1 + X0_1_2 = 1
 R2: X0_2_0 + X0_2_1 + X0_2_2 = 1
 R3: X1_0_0 + X1_0_1 + X1_0_2 = 1
 R4: X1_1_0 + X1_1_1 + X1_1_2 = 1
 R5: X1_2_0 + X1_2_1 + X1_2_2 = 1
 R6: X2_0_0 + X2_0_1 + X2_0_2 = 1
 R7: X2_1_0 + X2_1_1 + X2_1_2 = 1
 R8: X2_2_0 + X2_2_1 + X2_2_2 = 1
 R9: X0_0_0 + X0_1_0 + X0_2_0 = 1
 R10: X1_0_0 + X1_1_0 + X1_2_0 = 1
 R11: X2_0_0 + X2_1_0 + X2_2_0 = 1
 R12: X0_0_1 + X0_1_1 + X0_2_1 = 1
 R13: X1_0_1 + X1_1_1 + X1_2_1 = 1
 R14: X2_0_1 + X2_1_1 + X2_2_1 = 1
 R15: X0_0_2 + X0_1_2 + X0_2_2 = 1
 R16: X1_0_2 + X1_1_2 + X1_2_2 = 1
 R17: X2_0_2 + X2_1_2 + X2_2_2 = 1
 R18: - X0_0_0 + Y0_0_1_1_0 <= 0
 R19: - X0_0_0 + Y0_0_1_1_0 <= 0
 R20: - 2 X0_0_0 + Y0_0_1_1_0 >= -1
 R21: - X0_0_0 + Y0_0_1_2_0 <= 0
 R22: - X0_1_0 + Y0_0_1_2_0 <= 0
 R23: - X0_0_0 - X0_1_0 + Y0_0_1_2_0 >= -1
 R24: - X0_0_0 + Y0_0_2_1_0 <= 0
 R25: - X0_2_0 + Y0_0_2_1_0 <= 0
 R26: - X0_0_0 - X0_2_0 + Y0_0_2_1_0 >= -1
 R27: - X0_0_0 + Y0_0_2_2_0 <= 0
 R28: - X1_0_0 + Y0_0_2_2_0 <= 0
 R29: - X0_0_0 - X1_0_0 + Y0_0_2_2_0 >= -1
 R30: - X0_1_0 + Y0_1_1_2_0 <= 0
 R31: - X0_0_0 + Y0_1_1_2_0 <= 0
 R32: - X0_0_0 - X0_1_0 + Y0_1_1_2_0 >= -1
 R33: - X0_1_0 + Y0_1_2_2_0 <= 0
 R34: - X0_1_0 + Y0_1_2_2_0 <= 0
 R35: - 2 X0_1_0 + Y0_1_2_2_0 >= -1
 R36: - X0_2_0 + Y1_0_2_1_0 <= 0
 R37: - X0_0_0 + Y1_0_2_1_0 <= 0
 R38: - X0_0_0 - X0_2_0 + Y1_0_2_1_0 >= -1
 R39: - X0_2_0 + Y1_0_2_2_0 <= 0
 R40: - X0_1_0 + Y1_0_2_2_0 <= 0
 R41: - X0_1_0 - X0_2_0 + Y1_0_2_2_0 >= -1
 R42: - X1_0_0 + Y1_1_2_2_0 <= 0
 R43: - X0_0_0 + Y1_1_2_2_0 <= 0
 R44: - X0_0_0 - X1_0_0 + Y1_1_2_2_0 >= -1
 R45: - X0_0_1 + Y0_0_1_1_1 <= 0
 R46: - X0_0_1 + Y0_0_1_1_1 <= 0
 R47: - 2 X0_0_1 + Y0_0_1_1_1 >= -1
 R48: - X0_0_1 + Y0_0_1_2_1 <= 0
 R49: - X0_1_1 + Y0_0_1_2_1 <= 0
 R50: - X0_0_1 - X0_1_1 + Y0_0_1_2_1 >= -1
 R51: - X0_0_1 + Y0_0_2_1_1 <= 0
 R52: - X0_2_1 + Y0_0_2_1_1 <= 0
 R53: - X0_0_1 - X0_2_1 + Y0_0_2_1_1 >= -1
 R54: - X0_0_1 + Y0_0_2_2_1 <= 0
 R55: - X1_0_1 + Y0_0_2_2_1 <= 0
 R56: - X0_0_1 - X1_0_1 + Y0_0_2_2_1 >= -1
 R57: - X0_1_1 + Y0_1_1_2_1 <= 0
 R58: - X0_0_1 + Y0_1_1_2_1 <= 0
 R59: - X0_0_1 - X0_1_1 + Y0_1_1_2_1 >= -1
 R60: - X0_1_1 + Y0_1_2_2_1 <= 0
 R61: - X0_1_1 + Y0_1_2_2_1 <= 0
 R62: - 2 X0_1_1 + Y0_1_2_2_1 >= -1
 R63: - X0_2_1 + Y1_0_2_1_1 <= 0
 R64: - X0_0_1 + Y1_0_2_1_1 <= 0
 R65: - X0_0_1 - X0_2_1 + Y1_0_2_1_1 >= -1
 R66: - X0_2_1 + Y1_0_2_2_1 <= 0
 R67: - X0_1_1 + Y1_0_2_2_1 <= 0
 R68: - X0_1_1 - X0_2_1 + Y1_0_2_2_1 >= -1
 R69: - X1_0_1 + Y1_1_2_2_1 <= 0
 R70: - X0_0_1 + Y1_1_2_2_1 <= 0
 R71: - X0_0_1 - X1_0_1 + Y1_1_2_2_1 >= -1
 R72: - X0_0_2 + Y0_0_1_1_2 <= 0
 R73: - X0_0_2 + Y0_0_1_1_2 <= 0
 R74: - 2 X0_0_2 + Y0_0_1_1_2 >= -1
 R75: - X0_0_2 + Y0_0_1_2_2 <= 0
 R76: - X0_1_2 + Y0_0_1_2_2 <= 0
 R77: - X0_0_2 - X0_1_2 + Y0_0_1_2_2 >= -1
 R78: - X0_0_2 + Y0_0_2_1_2 <= 0
 R79: - X0_2_2 + Y0_0_2_1_2 <= 0
 R80: - X0_0_2 - X0_2_2 + Y0_0_2_1_2 >= -1
 R81: - X0_0_2 + Y0_0_2_2_2 <= 0
 R82: - X1_0_2 + Y0_0_2_2_2 <= 0
 R83: - X0_0_2 - X1_0_2 + Y0_0_2_2_2 >= -1
 R84: - X0_1_2 + Y0_1_1_2_2 <= 0
 R85: - X0_0_2 + Y0_1_1_2_2 <= 0
 R86: - X0_0_2 - X0_1_2 + Y0_1_1_2_2 >= -1
 R87: - X0_1_2 + Y0_1_2_2_2 <= 0
 R88: - X0_1_2 + Y0_1_2_2_2 <= 0
 R89: - 2 X0_1_2 + Y0_1_2_2_2 >= -1
 R90: - X0_2_2 + Y1_0_2_1_2 <= 0
 R91: - X0_0_2 + Y1_0_2_1_2 <= 0
 R92: - X0_0_2 - X0_2_2 + Y1_0_2_1_2 >= -1
 R93: - X0_2_2 + Y1_0_2_2_2 <= 0
 R94: - X0_1_2 + Y1_0_2_2_2 <= 0
 R95: - X0_1_2 - X0_2_2 + Y1_0_2_2_2 >= -1
 R96: - X1_0_2 + Y1_1_2_2_2 <= 0
 R97: - X0_0_2 + Y1_1_2_2_2 <= 0
 R98: - X0_0_2 - X1_0_2 + Y1_1_2_2_2 >= -1
Bounds
Binaries
 X0_0_0 X0_0_1 X0_0_2 X0_1_0 X0_1_1 X0_1_2 X0_2_0 X0_2_1 X0_2_2 X1_0_0
 X1_0_1 X1_0_2 X1_1_0 X1_1_1 X1_1_2 X1_2_0 X1_2_1 X1_2_2 X2_0_0 X2_0_1
 X2_0_2 X2_1_0 X2_1_1 X2_1_2 X2_2_0 X2_2_1 X2_2_2 Y0_0_1_1_0 Y0_0_1_1_1
 Y0_0_1_1_2 Y0_0_1_2_0 Y0_0_1_2_1 Y0_0_1_2_2 Y0_0_2_1_0 Y0_0_2_1_1
 Y0_0_2_1_2 Y0_0_2_2_0 Y0_0_2_2_1 Y0_0_2_2_2 Y0_1_1_2_0 Y0_1_1_2_1
 Y0_1_1_2_2 Y0_1_2_2_0 Y0_1_2_2_1 Y0_1_2_2_2 Y1_0_2_1_0 Y1_0_2_1_1
 Y1_0_2_1_2 Y1_0_2_2_0 Y1_0_2_2_1 Y1_0_2_2_2 Y1_1_2_2_0 Y1_1_2_2_1
 Y1_1_2_2_2
End
