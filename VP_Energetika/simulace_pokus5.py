# MODEL PRO KOMUNITNI ENERGETIKU
# VERZE: realna data, alokacni klic i preference zadany

import gurobipy as gu
from priprava_dat_2 import *
import numpy as np
import random


def pokus():

    krange = range(5)
    prange = range(3)

    # d_kp = np.array([[m.addVar(vtype=gu.GRB.CONTINUOUS, lb=0) for p in prange] for k in krange])

    # task: create predecessors for every kp pair

    kp_pairs = []
    print("k  p")
    for k in krange:
        for p in prange:
            print(k, p)
            kp_pairs.append((k, p))


    for i, pair in enumerate(kp_pairs):
        if i > 0:
            print(pair, " k: ", pair[0], " p: ", pair[1], " pred: ", kp_pairs[i - 1] )


def optimalizace(supply_it, demand_jt):

    nr_k = 2 # pocet_kol
    krange = range(nr_k)
    nr_p = 3
    prange = range(nr_p)

    # supply_i = 1000*supply_it[:, 0][0:3]
    # demand_j = 1000*1000*demand_jt[:, 0][0:2]
    supply_i = 1000*supply_it[:, 0][0:5]
    demand_j = 1000*1000*demand_jt[:, 0][0:7]


    nr_j = len(demand_j)
    jrange = range(nr_j)

    nr_i = len(supply_i)
    irange = range(nr_i)

    # pref_jip = np.zeros((nr_j, nr_i, nr_p), dtype=int)
    # pref_jip[0, 1, 0] = 1
    # pref_jip[0, 2, 1] = 1
    # pref_jip[1, 2, 0] = 1
    # pref_jip[1, 1, 1] = 1

    pref_jip = np.zeros((nr_j, nr_i, nr_p), dtype=int)
    pref_jip[0, 0, 0] = 1
    pref_jip[0, 4, 1] = 1
    pref_jip[0, 2, 2] = 1
    pref_jip[1, 4, 0] = 1
    pref_jip[1, 1, 1] = 1
    pref_jip[1, 0, 2] = 1
    pref_jip[2, 3, 0] = 1
    pref_jip[2, 0, 1] = 1
    pref_jip[2, 2, 2] = 1
    pref_jip[3, 3, 0] = 1
    pref_jip[3, 2, 1] = 1
    pref_jip[3, 1, 2] = 1
    pref_jip[4, 3, 0] = 1
    pref_jip[4, 2, 1] = 1
    pref_jip[4, 4, 2] = 1
    pref_jip[5, 4, 0] = 1
    pref_jip[5, 1, 1] = 1
    pref_jip[5, 2, 2] = 1
    pref_jip[6, 2, 0] = 1
    pref_jip[6, 1, 1] = 1
    pref_jip[6, 0, 2] = 1

    w_ij = np.array([[1./nr_j for j in jrange] for i in irange])


    kp_pairs = []
    print("k  p")
    for k in krange:
        for p in prange:
            print(k, p)
            kp_pairs.append((k, p))

    m = gu.Model("energetika")


    real_f_ij_kp = np.array([[[[m.addVar(vtype=gu.GRB.CONTINUOUS, lb=0) for p in prange] for k in krange] for j in jrange] for i in irange])

    possible_f_ij_kp = np.array([[[[m.addVar(vtype=gu.GRB.CONTINUOUS, lb=0) for k in prange] for k in krange] for j in jrange] for i in irange])

    d_jkp = np.array([[[m.addVar(vtype=gu.GRB.CONTINUOUS, lb=0) for p in prange] for k in krange] for j in jrange])

    s_ik = np.array([[m.addVar(vtype=gu.GRB.CONTINUOUS, lb=0) for k in krange] for i in irange])

    dT_j = np.array([m.addVar(vtype=gu.GRB.CONTINUOUS, lb=0) for j in jrange])
    sT_i = np.array([m.addVar(vtype=gu.GRB.CONTINUOUS, lb=0) for i in irange])


    m.update()

    for j in jrange:
        m.addConstr(d_jkp[j, 0, 0] == demand_j[j])

    for i in irange:
        m.addConstr(s_ik[i, 0] == supply_i[i])

    # constraints

    for j in jrange:
        for r, kp_pair in enumerate(kp_pairs):
            if r < len(kp_pairs) - 1:

                k, p = kp_pair
                k_next, p_next = kp_pairs[r + 1]
                m.addConstr(d_jkp[j, k_next, p_next] == d_jkp[j, k, p] - gu.quicksum(real_f_ij_kp[i, j, k, p] for i in irange))

            else:
                k, p = kp_pair
                m.addConstr(dT_j[j] == d_jkp[j, k, p] - gu.quicksum(real_f_ij_kp[i, j, k, p] for i in irange))

                for i in irange:
                    m.addConstr(sT_i[i] == s_ik[i, k] - gu.quicksum(real_f_ij_kp[i, j, k, p] for p in prange for j in jrange))

    for k in krange:
        if k < krange[-1]:
            for i in irange:
                m.addConstr(s_ik[i, k + 1] == s_ik[i, k] - gu.quicksum(real_f_ij_kp[i, j, k, p] for p in prange for j in jrange))


    for i in irange:
        for j in jrange:
            for k in krange:
                for p in prange:
                    m.addConstr(possible_f_ij_kp[i, j, k, p] == w_ij[i, j] * s_ik[i, k] * pref_jip[j, i, p])

    for i in irange:
        for j in jrange:

            for k in krange:
                for p in prange:
                    m.addConstr(real_f_ij_kp[i, j, k, p] == gu.min_(possible_f_ij_kp[i, j, k , p], d_jkp[j, k, p]))


    m.setObjective(gu.quicksum(dT_j[j] for j in jrange) + gu.quicksum(sT_i[i] for i in irange))
    m.optimize()


    for k in krange:
        for p in prange:
            for i in irange:
                for j in jrange:
                    if real_f_ij_kp[i, j, k, p].x > 0:
                        print("from: %i to %i\t" % (i, j), k, p, "flow : ", real_f_ij_kp[i, j, k, p].x)



    print("demand of energy")
    for k in krange:
        for j in jrange:
            print("kolo: ", k, " bod: ", i, d_jkp[j, k, 0].x)

    for j in jrange:
        print("kolo: ", "T", " bod: ", j, dT_j[j].x)

    print("")
    print("---")



    print("supply of energy")
    for k in krange:
        for i in irange:
            print("kolo: ", k, " bod: ", i, s_ik[i, k].x)

    for i in irange:
        print("kolo: ", "T", " bod: ", i, sT_i[i].x)

    print("")
    print("")



def control_panel():

    ind_from = 52
    ind_to = 54

    supply_it, demand_jt, spotreba_mista, produkce_mista, dates = read_and_send_data()
    supply_it = supply_it[:, ind_from:ind_to]
    demand_jt = demand_jt[:, ind_from:ind_to]
    dates = dates[ind_from:ind_to]


    nr_i, nr_t = np.shape(supply_it)

    nr_j, nr_t = np.shape(demand_jt)

    print("------------------------------------------")
    for t in range(len(dates)):
        print(t, dates[t], np.sum(supply_it[:, t]), np.sum(demand_jt[:, t]))


    irange = range(nr_i)
    jrange = range(nr_j)
    trange = range(nr_t)

    optimalizace(supply_it, demand_jt)


if __name__ == "__main__":
    control_panel()
    # pokus()
