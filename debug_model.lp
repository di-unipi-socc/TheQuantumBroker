\* General_Expr_Problem *\
Minimize
OBJ: 0.39 fake_fez_shots + 50 fake_fez_used + 0.195 fake_kyiv_shots
 + 450 fake_kyiv_used + 0.312 fake_marrakesh_shots + 425 fake_marrakesh_used
 + 0.39 fake_sherbrooke_shots + 425 fake_sherbrooke_used
 + 0.234 fake_torino_shots + 375 fake_torino_used
Subject To
constraint_0: 0.48 fake_fez_shots + 0.24 fake_kyiv_shots
 + 0.384 fake_marrakesh_shots + 0.48 fake_sherbrooke_shots
 + 0.288 fake_torino_shots <= 450
constraint_1: fake_torino_shots - 50 fake_torino_used >= 0
constraint_2: fake_kyiv_shots - 50 fake_kyiv_used >= 0
constraint_3: fake_sherbrooke_shots - 50 fake_sherbrooke_used >= 0
constraint_4: fake_fez_shots - 50 fake_fez_used >= 0
constraint_5: fake_marrakesh_shots - 50 fake_marrakesh_used >= 0
constraint_6: min_aux_0 >= 0.19
constraint_7: max_aux_1 <= 1500
constraint_8: fake_fez_shots + fake_kyiv_shots + fake_marrakesh_shots
 + fake_sherbrooke_shots + fake_torino_shots = 1024
constraint_9: fake_fez_used + fake_kyiv_used + fake_marrakesh_used
 + fake_sherbrooke_used + fake_torino_used >= 1
link_shots_nonnegative_fake_fez: fake_fez_shots >= 0
link_shots_nonnegative_fake_kyiv: fake_kyiv_shots >= 0
link_shots_nonnegative_fake_marrakesh: fake_marrakesh_shots >= 0
link_shots_nonnegative_fake_sherbrooke: fake_sherbrooke_shots >= 0
link_shots_nonnegative_fake_torino: fake_torino_shots >= 0
link_shots_used_upper_fake_fez: fake_fez_shots - 1024 fake_fez_used <= 0
link_shots_used_upper_fake_kyiv: fake_kyiv_shots - 1024 fake_kyiv_used <= 0
link_shots_used_upper_fake_marrakesh: fake_marrakesh_shots
 - 1024 fake_marrakesh_used <= 0
link_shots_used_upper_fake_sherbrooke: fake_sherbrooke_shots
 - 1024 fake_sherbrooke_used <= 0
link_shots_used_upper_fake_torino: fake_torino_shots - 1024 fake_torino_used
 <= 0
max_aux_1_ge_fake_fez: - 0.3 fake_fez_shots - 100 fake_fez_used + max_aux_1
 >= 0
max_aux_1_ge_fake_kyiv: - 0.15 fake_kyiv_shots - 900 fake_kyiv_used
 + max_aux_1 >= 0
max_aux_1_ge_fake_marrakesh: - 0.24 fake_marrakesh_shots
 - 850 fake_marrakesh_used + max_aux_1 >= 0
max_aux_1_ge_fake_sherbrooke: - 0.3 fake_sherbrooke_shots
 - 850 fake_sherbrooke_used + max_aux_1 >= 0
max_aux_1_ge_fake_torino: - 0.18 fake_torino_shots - 750 fake_torino_used
 + max_aux_1 >= 0
min_aux_0_le_fake_fez: 999999.053 fake_fez_used + min_aux_0 <= 1000000
min_aux_0_le_fake_kyiv: 999999.0588 fake_kyiv_used + min_aux_0 <= 1000000
min_aux_0_le_fake_marrakesh: 999999.0297 fake_marrakesh_used + min_aux_0
 <= 1000000
min_aux_0_le_fake_sherbrooke: 999999.0472 fake_sherbrooke_used + min_aux_0
 <= 1000000
min_aux_0_le_fake_torino: 999999.0443 fake_torino_used + min_aux_0 <= 1000000
total_shots: fake_fez_shots + fake_kyiv_shots + fake_marrakesh_shots
 + fake_sherbrooke_shots + fake_torino_shots = 1024
Bounds
 0 <= fake_fez_shots
 0 <= fake_kyiv_shots
 0 <= fake_marrakesh_shots
 0 <= fake_sherbrooke_shots
 0 <= fake_torino_shots
 max_aux_1 free
 min_aux_0 free
Generals
fake_fez_shots
fake_kyiv_shots
fake_marrakesh_shots
fake_sherbrooke_shots
fake_torino_shots
Binaries
fake_fez_used
fake_kyiv_used
fake_marrakesh_used
fake_sherbrooke_used
fake_torino_used
End
