import os
os.environ["MODIN_ENGINE"] = "dask"
import modin.config as modin_cfg
modin_cfg.Engine.put("dask")
import modin.pandas as pd
import numpy as np
import pandas
import time

pilot_study = False


def main():
    global_data = []
    trials = 10 if pilot_study else 1000
    datasets = ["circles.csv",  "tripadvisor.csv", "taxi.csv"]
    for dataset in datasets:
        print("--------- Dataset: {} ----------".format(dataset))
        modin_df = pd.read_csv(dataset)
        local_data = []
        for i in range(trials):
            pandas_duration_read = 0.0
            start = time.time()
            pandas_df = pandas.read_csv(dataset)
            end = time.time()
            pandas_duration_read += end - start

            #pandas_duration_read /= trials
            print("Time to read with pandas: {} seconds".format(round(pandas_duration_read, 5)))

            modin_duration_read = 0.0
            start = time.time()
            modin_df = pd.read_csv(dataset)
            end = time.time()
            modin_duration_read += end - start

            #modin_duration_read /= trials
            print("Time to read with Modin: {} seconds".format(round(modin_duration_read, 5)))
            print("## Modin is {}x faster than pandas at `read_csv`!".format(round(pandas_duration_read / modin_duration_read, 2)))

            # -----------------------------------------------------------------------------------------------------------------

            concat_times = 5    
            pandas_duration_concat = 0.0
            start = time.time()
            big_pandas_df = pandas.concat([pandas_df for _ in range(concat_times)])
            end = time.time()
            pandas_duration_concat += end - start

            #pandas_duration_concat /= trials
            print("Time to concat with pandas: {} seconds".format(round(pandas_duration_concat, 5)))

            modin_duration_concat = 0.0
            start = time.time()
            big_modin_df = pd.concat([modin_df for _ in range(concat_times)])
            end = time.time()
            modin_duration_concat += end - start

            #modin_duration_concat /= trials
            print("Time to concat with Modin: {} seconds".format(round(modin_duration_concat, 5)))
            print("Modin is {}x faster than pandas at `concat`!".format(round(pandas_duration_concat / modin_duration_concat, 2)))

            # -----------------------------------------------------------------------------------------------------------------

            pandas_duration_apply = 0.0
            start = time.time()
            rounded_trip_distance_pandas = big_pandas_df.dropna()
            end = time.time()
            pandas_duration_apply += end - start

            #pandas_duration_apply /= trials
            print("Time to apply with pandas: {} seconds".format(round(pandas_duration_apply, 5)))

            modin_duration_apply = 0.0
            start = time.time()
            rounded_trip_distance_modin = big_modin_df.dropna()
            end = time.time()
            modin_duration_apply += end - start

            #modin_duration_apply /= trials
            print("Time to apply with Modin: {} seconds".format(round(modin_duration_apply, 5)))
            print("Modin is {}x faster than pandas at `apply` on one column!".format(round(pandas_duration_apply / modin_duration_apply, 2)))

            local_data.append([pandas_duration_read, modin_duration_read, pandas_duration_concat, modin_duration_concat, pandas_duration_apply, modin_duration_apply])
            del pandas_df
            del modin_df
            del big_pandas_df
            del big_modin_df
        global_data.append(local_data)
    with open('pilot_study.npy' if pilot_study else 'data.npy', 'wb') as f:
        np.save(f, global_data)
        print("Saved data!")

if __name__ == '__main__':
    main()
