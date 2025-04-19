# FUNGUJICI SIMULACE PRO PEVNE NAGENEROVANE PREFERENCE A


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


def simulace(supply_it, demand_jt):

    random.seed(21)

    supply_i = 1000*supply_it[:, 0][0:5]
    demand_j = 1000*1000*demand_jt[:, 0][0:7]

    nr_k = 2 # pocet_kol
    krange = range(nr_k)

    nr_p = 3 # maximalni dovoleny pocet preferenci
    prange = range(nr_p)

    nr_i = np.shape(supply_i)[0]
    nr_j = np.shape(demand_j)[0]

    jrange = range(nr_j)
    irange = range(nr_i)

    preference_indexes_j = []
    for j in jrange:
        preference = list(irange).copy()
        random.shuffle(preference)
        preference_indexes_j.append(preference)
        print(j, " : ", preference)

    print("-----------------------------------------")
    print("-----------------------------------------")
    np.set_printoptions(precision=1, suppress=True)
    print("supply_i: ", supply_i, np.sum(supply_i))
    print("demand_j: ", demand_j, np.sum(demand_j))
    print("-----------------------------------------")

    w_ij = np.array([[1./nr_j for j in jrange] for i in irange])


    for i in irange:
        print(supply_i[i], np.dot(w_ij[i], supply_i[i]))


    for k in krange:
        odecist_i = np.zeros_like(supply_i)
        print("kolo: ", k)
        print("supply: ")
        for i in irange:
            print(supply_i[i], np.dot(w_ij[i], supply_i[i]))

        print("---------")

        for j in jrange:
            for p in prange:
                i = preference_indexes_j[j][p]
                possible_flow_ij = w_ij[i, j] * supply_i[i]
                flow_ij = min(possible_flow_ij, demand_j[j])
                print("possible  from %i to %i: " %(i, j), possible_flow_ij, "\t", "real: ", flow_ij)


                odecist_i[i] += flow_ij
                demand_j[j] -= flow_ij
            print("odecist: ", odecist_i)
            print("-----------------------------------------")

        supply_i -= odecist_i

        print("%i demand: " % j, demand_j[j], "\n")
        print("supply: ")
        for i in irange:
            print(supply_i[i])

        print("")
        print("demand: ")
        for j in jrange:
            print(demand_j[j])
        print("\n\n")


        print("-----------------------------------------")


    for j in jrange:
        for p in prange:
            i = preference_indexes_j[j][p]
            print("pref_jip[%i, %i, %i] = 1" % (j, i, p))


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

    simulace(supply_it, demand_jt)
    # optimalizace(supply_it, demand_jt)



if __name__ == "__main__":
    control_panel()
    # pokus()
