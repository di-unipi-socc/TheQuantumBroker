\* General_Expr_Problem *\
Minimize
OBJ: min_aux_1
Subject To
constraint_0: 0.238823694921 fake_marrakesh_shots <= 15000
constraint_1: fake_marrakesh_shots - 100 fake_marrakesh_used >= 0
constraint_2: max_aux_0 <= 80000
constraint_3: fake_marrakesh_shots = 4096
constraint_4: fake_marrakesh_used >= 1
link_shots_nonnegative_fake_marrakesh: fake_marrakesh_shots >= 0
link_shots_used_upper_fake_marrakesh: fake_marrakesh_shots
 - 4096 fake_marrakesh_used <= 0
max_aux_0_ge_fake_marrakesh: - 0.44 fake_marrakesh_shots
 - 85 fake_marrakesh_used + max_aux_0 >= 0
min_aux_1_le_fake_marrakesh: - 0.9494 fake_marrakesh_used + min_aux_1 <= 0
total_shots: fake_marrakesh_shots = 4096
Bounds
 0 <= fake_marrakesh_shots
 max_aux_0 free
 min_aux_1 free
Generals
fake_marrakesh_shots
Binaries
fake_marrakesh_used
End
