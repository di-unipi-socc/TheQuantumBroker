\* General_Expr_Problem *\
Minimize
OBJ: min_aux_0
Subject To
constraint_0: 0.390214830293 fake_fez_shots
 + 0.238823694921 fake_marrakesh_shots + 0.12059168563 fake_sherbrooke_shots
 + 0.349867618919 fake_torino_shots <= 15000
constraint_1: fake_torino_shots - 100 fake_torino_used >= 0
constraint_2: fake_sherbrooke_shots - 100 fake_sherbrooke_used >= 0
constraint_3: fake_fez_shots - 100 fake_fez_used >= 0
constraint_4: fake_marrakesh_shots - 100 fake_marrakesh_used >= 0
constraint_5: min_aux_0 >= 0.9
constraint_6: max_aux_1 <= 80000
constraint_7: fake_fez_shots + fake_marrakesh_shots + fake_sherbrooke_shots
 + fake_torino_shots = 4096
constraint_8: fake_fez_used + fake_marrakesh_used + fake_sherbrooke_used
 + fake_torino_used >= 1
link_shots_nonnegative_fake_fez: fake_fez_shots >= 0
link_shots_nonnegative_fake_marrakesh: fake_marrakesh_shots >= 0
link_shots_nonnegative_fake_sherbrooke: fake_sherbrooke_shots >= 0
link_shots_nonnegative_fake_torino: fake_torino_shots >= 0
link_shots_used_upper_fake_fez: fake_fez_shots - 4096 fake_fez_used <= 0
link_shots_used_upper_fake_marrakesh: fake_marrakesh_shots
 - 4096 fake_marrakesh_used <= 0
link_shots_used_upper_fake_sherbrooke: fake_sherbrooke_shots
 - 4096 fake_sherbrooke_used <= 0
link_shots_used_upper_fake_torino: fake_torino_shots - 4096 fake_torino_used
 <= 0
max_aux_1_ge_fake_fez: - 0.62 fake_fez_shots - 30 fake_fez_used + max_aux_1
 >= 0
max_aux_1_ge_fake_marrakesh: - 0.44 fake_marrakesh_shots
 - 85 fake_marrakesh_used + max_aux_1 >= 0
max_aux_1_ge_fake_sherbrooke: - 0.57 fake_sherbrooke_shots
 - 85 fake_sherbrooke_used + max_aux_1 >= 0
max_aux_1_ge_fake_torino: - 0.35 fake_torino_shots - 75 fake_torino_used
 + max_aux_1 >= 0
min_aux_0_le_fake_fez: - 0.9097 fake_fez_used + min_aux_0 <= 0
min_aux_0_le_fake_marrakesh: - 0.9494 fake_marrakesh_used + min_aux_0 <= 0
min_aux_0_le_fake_sherbrooke: - 0.9196 fake_sherbrooke_used + min_aux_0 <= 0
min_aux_0_le_fake_torino: - 0.9246 fake_torino_used + min_aux_0 <= 0
total_shots: fake_fez_shots + fake_marrakesh_shots + fake_sherbrooke_shots
 + fake_torino_shots = 4096
Bounds
 0 <= fake_fez_shots
 0 <= fake_marrakesh_shots
 0 <= fake_sherbrooke_shots
 0 <= fake_torino_shots
 max_aux_1 free
 min_aux_0 free
Generals
fake_fez_shots
fake_marrakesh_shots
fake_sherbrooke_shots
fake_torino_shots
Binaries
fake_fez_used
fake_marrakesh_used
fake_sherbrooke_used
fake_torino_used
End
