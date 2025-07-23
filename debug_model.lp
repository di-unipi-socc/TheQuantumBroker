\* General_Expr_Problem *\
Minimize
OBJ: 3.7 fake_kyiv_shots + 2.9 fake_kyoto_shots + 3.2 fake_sherbrooke_shots
 + 3.2 fake_singapore_shots + 3.5 fake_torino_shots
Subject To
constraint_0: 3.7 fake_kyiv_shots + 2.9 fake_kyoto_shots
 + 3.2 fake_sherbrooke_shots + 3.2 fake_singapore_shots
 + 3.5 fake_torino_shots <= 3600
constraint_1: fake_torino_shots >= 100
constraint_2: min_aux_0 >= 0.8
constraint_3: fake_kyiv_shots + fake_kyoto_shots + fake_sherbrooke_shots
 + fake_singapore_shots + fake_torino_shots = 1024
constraint_4: fake_kyiv_used + fake_kyoto_used + fake_sherbrooke_used
 + fake_singapore_used + fake_torino_used >= 1
link_shots_nonnegative_fake_kyiv: fake_kyiv_shots >= 0
link_shots_nonnegative_fake_kyoto: fake_kyoto_shots >= 0
link_shots_nonnegative_fake_sherbrooke: fake_sherbrooke_shots >= 0
link_shots_nonnegative_fake_singapore: fake_singapore_shots >= 0
link_shots_nonnegative_fake_torino: fake_torino_shots >= 0
link_shots_used_upper_fake_kyiv: fake_kyiv_shots - 1024 fake_kyiv_used <= 0
link_shots_used_upper_fake_kyoto: fake_kyoto_shots - 1024 fake_kyoto_used <= 0
link_shots_used_upper_fake_sherbrooke: fake_sherbrooke_shots
 - 1024 fake_sherbrooke_used <= 0
link_shots_used_upper_fake_singapore: fake_singapore_shots
 - 1024 fake_singapore_used <= 0
link_shots_used_upper_fake_torino: fake_torino_shots - 1024 fake_torino_used
 <= 0
min_aux_0_le_fake_kyiv: 999999.05 fake_kyiv_used + min_aux_0 <= 1000000
min_aux_0_le_fake_kyoto: 999999.1 fake_kyoto_used + min_aux_0 <= 1000000
min_aux_0_le_fake_sherbrooke: 999999.15 fake_sherbrooke_used + min_aux_0
 <= 1000000
min_aux_0_le_fake_singapore: 999999.1 fake_singapore_used + min_aux_0
 <= 1000000
min_aux_0_le_fake_torino: 999999.1 fake_torino_used + min_aux_0 <= 1000000
total_shots: fake_kyiv_shots + fake_kyoto_shots + fake_sherbrooke_shots
 + fake_singapore_shots + fake_torino_shots = 1024
Bounds
 0 <= fake_kyiv_shots
 0 <= fake_kyoto_shots
 0 <= fake_sherbrooke_shots
 0 <= fake_singapore_shots
 0 <= fake_torino_shots
 min_aux_0 free
Generals
fake_kyiv_shots
fake_kyoto_shots
fake_sherbrooke_shots
fake_singapore_shots
fake_torino_shots
Binaries
fake_kyiv_used
fake_kyoto_used
fake_sherbrooke_used
fake_singapore_used
fake_torino_used
End
