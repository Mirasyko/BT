\ Model EnergyRedistribution
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x3584a54a4bf47f6
Minimize
  S_0_0_3 + S_0_1_3 + D_0_0_3 + D_0_1_3 + D_0_2_3
Subject To
 weight_sum_0: w_0_0 + w_0_1 + w_0_2 = 1
 weight_sum_1: w_1_0 + w_1_1 + w_1_2 = 1
 flow_supply_0_0_0_1: - 50 w_0_0 + f_0_0_0_1 <= 0
 flow_demand_0_0_0_1: f_0_0_0_1 <= 40
 flow_supply_0_0_1_1: - 50 w_0_1 + f_0_0_1_1 <= 0
 flow_demand_0_0_1_1: f_0_0_1_1 <= 50
 flow_supply_0_0_2_1: - 50 w_0_2 + f_0_0_2_1 <= 0
 flow_demand_0_0_2_1: f_0_0_2_1 <= 30
 flow_supply_0_1_0_1: - 60 w_1_0 + f_0_1_0_1 <= 0
 flow_demand_0_1_0_1: f_0_1_0_1 <= 40
 flow_supply_0_1_1_1: - 60 w_1_1 + f_0_1_1_1 <= 0
 flow_demand_0_1_1_1: f_0_1_1_1 <= 50
 flow_supply_0_1_2_1: - 60 w_1_2 + f_0_1_2_1 <= 0
 flow_demand_0_1_2_1: f_0_1_2_1 <= 30
 update_supply_0_0_1: S_0_0_1 + f_0_0_0_1 + f_0_0_1_1 + f_0_0_2_1 = 50
 update_supply_0_1_1: S_0_1_1 + f_0_1_0_1 + f_0_1_1_1 + f_0_1_2_1 = 60
 update_demand_0_0_1: D_0_0_1 + f_0_0_0_1 + f_0_1_0_1 = 40
 update_demand_0_1_1: D_0_1_1 + f_0_0_1_1 + f_0_1_1_1 = 50
 update_demand_0_2_1: D_0_2_1 + f_0_0_2_1 + f_0_1_2_1 = 30
 flow_demand_0_0_0_2: - D_0_0_1 + f_0_0_0_2 <= 0
 flow_demand_0_0_1_2: - D_0_1_1 + f_0_0_1_2 <= 0
 flow_demand_0_0_2_2: - D_0_2_1 + f_0_0_2_2 <= 0
 flow_demand_0_1_0_2: - D_0_0_1 + f_0_1_0_2 <= 0
 flow_demand_0_1_1_2: - D_0_1_1 + f_0_1_1_2 <= 0
 flow_demand_0_1_2_2: - D_0_2_1 + f_0_1_2_2 <= 0
 update_supply_0_0_2: - S_0_0_1 + S_0_0_2 + f_0_0_0_2 + f_0_0_1_2
   + f_0_0_2_2 = 0
 update_supply_0_1_2: - S_0_1_1 + S_0_1_2 + f_0_1_0_2 + f_0_1_1_2
   + f_0_1_2_2 = 0
 update_demand_0_0_2: - D_0_0_1 + D_0_0_2 + f_0_0_0_2 + f_0_1_0_2 = 0
 update_demand_0_1_2: - D_0_1_1 + D_0_1_2 + f_0_0_1_2 + f_0_1_1_2 = 0
 update_demand_0_2_2: - D_0_2_1 + D_0_2_2 + f_0_0_2_2 + f_0_1_2_2 = 0
 flow_demand_0_0_0_3: - D_0_0_2 + f_0_0_0_3 <= 0
 flow_demand_0_0_1_3: - D_0_1_2 + f_0_0_1_3 <= 0
 flow_demand_0_0_2_3: - D_0_2_2 + f_0_0_2_3 <= 0
 flow_demand_0_1_0_3: - D_0_0_2 + f_0_1_0_3 <= 0
 flow_demand_0_1_1_3: - D_0_1_2 + f_0_1_1_3 <= 0
 flow_demand_0_1_2_3: - D_0_2_2 + f_0_1_2_3 <= 0
 update_supply_0_0_3: - S_0_0_2 + S_0_0_3 + f_0_0_0_3 + f_0_0_1_3
   + f_0_0_2_3 = 0
 update_supply_0_1_3: - S_0_1_2 + S_0_1_3 + f_0_1_0_3 + f_0_1_1_3
   + f_0_1_2_3 = 0
 update_demand_0_0_3: - D_0_0_2 + D_0_0_3 + f_0_0_0_3 + f_0_1_0_3 = 0
 update_demand_0_1_3: - D_0_1_2 + D_0_1_3 + f_0_0_1_3 + f_0_1_1_3 = 0
 update_demand_0_2_3: - D_0_2_2 + D_0_2_3 + f_0_0_2_3 + f_0_1_2_3 = 0
 flow_supply_0_0_0_2: f_0_0_0_2 + [ - w_0_0 * S_0_0_1 ] <= 0
 flow_supply_0_0_1_2: f_0_0_1_2 + [ - w_0_1 * S_0_0_1 ] <= 0
 flow_supply_0_0_2_2: f_0_0_2_2 + [ - w_0_2 * S_0_0_1 ] <= 0
 flow_supply_0_1_0_2: f_0_1_0_2 + [ - w_1_0 * S_0_1_1 ] <= 0
 flow_supply_0_1_1_2: f_0_1_1_2 + [ - w_1_1 * S_0_1_1 ] <= 0
 flow_supply_0_1_2_2: f_0_1_2_2 + [ - w_1_2 * S_0_1_1 ] <= 0
 flow_supply_0_0_0_3: f_0_0_0_3 + [ - w_0_0 * S_0_0_2 ] <= 0
 flow_supply_0_0_1_3: f_0_0_1_3 + [ - w_0_1 * S_0_0_2 ] <= 0
 flow_supply_0_0_2_3: f_0_0_2_3 + [ - w_0_2 * S_0_0_2 ] <= 0
 flow_supply_0_1_0_3: f_0_1_0_3 + [ - w_1_0 * S_0_1_2 ] <= 0
 flow_supply_0_1_1_3: f_0_1_1_3 + [ - w_1_1 * S_0_1_2 ] <= 0
 flow_supply_0_1_2_3: f_0_1_2_3 + [ - w_1_2 * S_0_1_2 ] <= 0
Bounds
 w_0_0 <= 1
 w_0_1 <= 1
 w_0_2 <= 1
 w_1_0 <= 1
 w_1_1 <= 1
 w_1_2 <= 1
End
