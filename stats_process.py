import pickle
import sys
from Join_scheme.data_prepare import process_stats_data

def generate_and_save_buckets():
    data_path = "./data/stats_simplified"
    model_folder = "./models/"
    n_bins=500
    bucket_method = "greedy"
    save_bucket_bins = True
    actual_data = None
    data, null_values, key_attrs, table_buckets, equivalent_keys, schema, bin_size = process_stats_data(data_path,
                                        model_folder, n_bins, bucket_method, save_bucket_bins, actual_data=actual_data)
    for key in table_buckets:
        print(f"{key}: ")
        table_buckets[key].output()
    
    with open(model_folder+"table_buckets.pkl", "wb") as tb:
        pickle.dump(table_buckets, tb, pickle.HIGHEST_PROTOCOL)

def check_opt_buckets():
    opt_buckets_path="./models/buckets.pkl"
    with open(opt_buckets_path, "rb") as bf:
        opt_buckets=pickle.load(bf)
    
    for key in opt_buckets:
        print(f"{key}:")
        opt_buckets[key].output()

def check_table_buckets():
    table_buckets_path="./models/table_buckets.pkl"
    with open(table_buckets_path, "rb") as tbp:
        table_buckets=pickle.load(tbp)
    
    for key in table_buckets:
        print(f"{key}:")
        table_buckets[key].output()

if __name__ == "__main__":
    # generate_and_save_buckets()
    with open("output.txt", "w") as f:
        original_stdout=sys.stdout
        sys.stdout=f
    
        check_opt_buckets()
        check_table_buckets()
