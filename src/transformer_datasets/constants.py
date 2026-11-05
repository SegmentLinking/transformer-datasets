SIM_VARS = ["sim_pt", "sim_eta", "sim_phi", "sim_q", "sim_vx", "sim_vy", "sim_vz", "sim_vtxperp"]

T3_VARS = ["t3_pt", "t3_eta", "t3_phi"]
T3_HIT_SIMIDX = ["t3_hit_0_simIdx", "t3_hit_1_simIdx", "t3_hit_2_simIdx", "t3_hit_3_simIdx", "t3_hit_4_simIdx", "t3_hit_5_simIdx"]
T3_HIT_LAYER = ["t3_hit_0_layer", "t3_hit_1_layer", "t3_hit_2_layer", "t3_hit_3_layer", "t3_hit_4_layer", "t3_hit_5_layer"]
T3_PMATCHED = ["t3_pMatched"]
LS_INDEX = ["t3_lsIdx0", "t3_lsIdx1"]

LS_VARS = ["ls_dPhis", "ls_dPhiChanges", "ls_dAlphaInners", "ls_dAlphaOuters", "ls_dAlphaInnerOuters"]

MD_VARS = ["md_anchor_x", "md_anchor_y", "md_anchor_z", "md_other_x", "md_other_y", "md_other_z", "md_dphi", "md_dphichange", "md_dz"]
MD_INDEX = ["ls_mdIdx0", "ls_mdIdx1"]

TARGET = ["t3_simIdx"]

FAKE_TARGET = ["t3_isFake"]

LAYER_INFO = ["md_layer"]
MD_SIMIDX = ["md_simIdx"]

ALL_COLUMNS = SIM_VARS + T3_VARS + T3_HIT_SIMIDX + T3_HIT_LAYER + T3_PMATCHED + LS_INDEX + LS_VARS + MD_VARS + MD_INDEX + TARGET + FAKE_TARGET + LAYER_INFO + MD_SIMIDX

UNMATCHED = -999

MIN_T3_PT = 0
MAX_T3_PT = 2000
