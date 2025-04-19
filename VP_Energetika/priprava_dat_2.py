import pandas as pd
import numpy as np


def read_and_send_data(print_read=True):

    demand_jt = []
    supply_it = []
    dates = []
    ind_from = 100
    ind_to = 155

    spotreba_mista = ["radnice", "zs_preislerova", "zs_komenskeho", "ms_preislerova", "ms_pod_homolkou", "ms_vrchlickeho", "dum_pro_duchodce", "ms_drasarova",
                     "ms_tovarni", "ms_na_machovne", "zimni_stad", "plavecky_areal", "parkovaci_dum"]

    produkce_mista = ["zs_preislerova", "zs_komenskeho", "zimni_stad", "pristavba_preislerova", "parkovaci_dum",
                      "ms_vrchlickeho", "ms_preislerova", "ms_pod_homolkou", "ms_na_machovne", "ms_drasarova"]

    irange = range(len(produkce_mista))
    jrange = range(len(supply_it))

    base_df = pd.read_csv("beroun_counsumptions.csv")

    base_df['timestamp'] = pd.to_datetime(base_df['timestamp'])  # Ensure timestamp is datetime

    year = 2023
    month = 5
    filtered_spotreba_df = base_df[(base_df['timestamp'].dt.year == year) & (base_df['timestamp'].dt.month == month)]


    for j, sp in enumerate(spotreba_mista):

        spotreba = filtered_spotreba_df[sp].values.tolist()
        dates = filtered_spotreba_df["timestamp"].values.tolist()

        if print_read:
            print(len(spotreba), sp)

        demand_jt.append([])
        demand_jt[j] = spotreba[ind_from: ind_to]


    # print("-------------------------------------------------------------------------")

    for i, prod in enumerate(produkce_mista):

        temp_df = pd.read_csv("produkce/production_%s.csv" % prod)  # Read the next file
        temp_df['timestamp2'] = pd.to_datetime(temp_df['timestamp'])  # Ensure timestamp is datetime
        column_names = temp_df.columns
        col_name = "prod"
        if "prod" not in column_names:
            col_name = "prod_total"


        year = 2023
        month = 5
        filtered_produkce_df = temp_df[(temp_df['timestamp2'].dt.year == year) & (temp_df['timestamp2'].dt.month == month)]

        produkce = filtered_produkce_df[col_name].values.tolist()


        dates_prod = filtered_produkce_df["timestamp"]
        supply_it.append([])
        supply_it[i] = produkce[ind_from:ind_to]

        dates = dates_prod[ind_from:ind_to].values.tolist()

    supply_it = np.array(supply_it)
    demand_jt = np.array(demand_jt)


    if print_read:
        print(supply_it.shape)
        print(demand_jt.shape)
        print(len(dates))
        print(dates)


        for t in range(len(dates)):
            print(t, dates[t], np.sum(supply_it[:, t]), np.sum(demand_jt[:, t]))


    return supply_it, demand_jt, spotreba_mista, produkce_mista, dates

if __name__ == "__main__":

    read_and_send_data()
