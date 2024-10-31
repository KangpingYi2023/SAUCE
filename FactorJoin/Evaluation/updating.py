import numpy as np
import pickle
import time
import os
import psycopg2
import copy

import sys
sys.path.append("./")
from Schemas.stats.schema import gen_stats_light_schema
from Evaluation.training import train_one_stats, test_trained_BN_on_stats
from Join_scheme.data_prepare import read_table_csv, update_stats_data, update_stats_data_online
from BayesCard.Models.Bayescard_BN import Bayescard_BN


def timestamp_transorform(time_string, start_date="2010-07-19 00:00:00"):
    start_date_int = time.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    time_array = time.strptime(time_string, "%Y-%m-%d %H:%M:%S")
    return int(time.mktime(time_array)) - int(time.mktime(start_date_int))


def get_data_by_date(data_path, time_date="2014-01-01 00:00:00"):
    time_value = timestamp_transorform(time_date)
    if not data_path.endswith(".csv"):
        data_path += "/{}.csv"
    schema = gen_stats_light_schema(data_path)
    before_data = dict()
    after_data = dict()
    for table_obj in schema.tables:
        table_name = table_obj.table_name
        df_rows = read_table_csv(table_obj, db_name="stats")

        idx = len(df_rows)
        for attribute in df_rows.columns:
            if "Date" in attribute:
                idx = np.searchsorted(df_rows[attribute].values, time_value)
                break

        before_data[table_name] = df_rows[:idx] if idx > 0 else None
        after_data[table_name] = df_rows[idx:] if idx < len(df_rows) else None
    return before_data, after_data

def get_update_data_by_date(data_path, processed_data=None, time_date="2014-01-01 00:00:00"):
    time_value = timestamp_transorform(time_date)
    if not data_path.endswith(".csv"):
        data_path += "/{}.csv"
    schema = gen_stats_light_schema(data_path)

    after_data = dict()
    after_data_processed = dict()
    for table_obj in schema.tables:
        table_name = table_obj.table_name
        df_rows = read_table_csv(table_obj, db_name="stats")
        if processed_data is not None:
            df_rows_choice = processed_data[table_name]
        else:
            df_rows_choice = df_rows

        idx = len(df_rows_choice)
        for attribute in df_rows_choice.columns:
            if "Date" in attribute:
                idx = np.searchsorted(df_rows_choice[attribute].values, time_value)
                break

        after_data[table_name] = df_rows[idx:] if idx < len(df_rows) else None
        after_data_processed[table_name] = df_rows_choice[idx:] if idx < len(df_rows) else None
    return after_data, after_data_processed

def update_one_stats(FJmodel, buckets, table_buckets, data_path, save_model_folder, save_bucket_bins=False,
                     update_BN=True, retrain_BN=False, old_data=None, validate=False):
    """
    Incrementally update the FactorJoin model
    """
    data, table_buckets, null_values = update_stats_data(data_path, save_model_folder, buckets, table_buckets,
                                                         save_bucket_bins)
    FJmodel.table_buckets = table_buckets
    if update_BN:
        # updating the single table estimator
        if retrain_BN:
            # retrain the BN based on the new and old data
            for table in FJmodel.schema.tables:
                t_name = table.table_name
                if t_name in data and data[t_name] is not None:
                    bn = Bayescard_BN(t_name, table_buckets[t_name].id_attributes, table_buckets[t_name].bin_sizes,
                                      null_values=null_values[t_name])
                    new_data = old_data[t_name].append(data[t_name], ignore_index=True)
                    bn.build_from_data(new_data)
                    if validate:
                        test_trained_BN_on_stats(bn, t_name)
                    FJmodel.bns[t_name] = bn
        else:
            # incrementally update BN
            for table in FJmodel.schema.tables:
                t_name = table.table_name
                if t_name in data and data[t_name] is not None:
                    bn = FJmodel.bns[t_name]
                    bn.null_values = null_values[t_name]
                    bn.update_from_data(data)

    model_path = os.path.join(save_model_folder, f"update_model.pkl")
    pickle.dump(FJmodel, open(model_path, 'wb'), pickle.HIGHEST_PROTOCOL)
    print(f"models save at {model_path}")

def update_one_imdb():
    "TODO: implement the update on IMDB, should be straight-forward as it uses the sampling for base-table"
    return

def eval_update(data_folder, model_path, n_dim_dist, bin_size, bucket_method, split_date="2014-01-01 00:00:00", seed=0):
    np.random.seed(seed)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    before_data, after_data = get_data_by_date(data_folder, split_date)
    print("************************************************************")
    print(f"Training the model with data before {split_date}")
    start_time = time.time()
    train_one_stats("stats", data_folder, model_path, n_dim_dist, bin_size, bucket_method, True, actual_data=before_data)
    print(f"training completed, took {time.time() - start_time} sec")

    # loading the trained model and buckets
    with open(os.path.join(model_path, "buckets.pkl"), "rb") as f:
        buckets = pickle.load(f)
    with open(os.path.join(model_path, f"model_stats_{bucket_method}_{bin_size}.pkl"), "rb") as f:
        FJmodel = pickle.load(f)
    print("************************************************************")
    print(f"Updating the model with data after {split_date}")
    start_time = time.time()
    table_buckets = FJmodel.table_buckets
    null_values = FJmodel.null_value
    data, table_buckets, null_values = update_stats_data(data_folder, model_path, buckets, table_buckets,
                                                         null_values, False, after_data)
    for table in FJmodel.schema.tables:
        t_name = table.table_name
        if t_name in data and data[t_name] is not None:
            bn = FJmodel.bns[t_name]
            bn.null_values = null_values[t_name]
            bn.update_from_data(data[t_name])
    print(f"updating completed, took {time.time() - start_time} sec")
    model_path = os.path.join(model_path, f"updated_model_stats_{bucket_method}_{bin_size}.pkl")
    pickle.dump(FJmodel, open(model_path, 'wb'), pickle.HIGHEST_PROTOCOL)
    print(f"updated models save at {model_path}")

def init_postgre(dataset, data_path):
    conn_params = {
        'dbname': dataset,
        'user': 'kangping',
        'password': 'kangping',
        'host': 'localhost',
        'port': '5432'
    }

    try:
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()
        conn.autocommit = False 
    except Exception as e:
        print(f"error: {e}")
        conn.rollback()

    table_list=["badges", "comments", "users", "tags", "posts", "votes", "posthistory", "postlinks"]
    
    if not data_path.endswith(".csv"):
        data_path += "/{}.csv"

    for table_name in table_list:
        csv_path=data_path.format(table_name)

def read_origin_data(data_path):
    if not data_path.endswith(".csv"):
        data_path += "/{}.csv"
    schema = gen_stats_light_schema(data_path)
    origin_data = dict()
    processed_data=dict()
    for table_obj in schema.tables:
        table_name = table_obj.table_name
        df_rows = read_table_csv(table_obj, db_name="stats")
        
        origin_data[table_name]=df_rows

        df_rows_tmp=copy.deepcopy(df_rows)
        for attribute in df_rows.columns:
            if "Date" in attribute:
                if df_rows[attribute].values.dtype == 'object':
                    new_value = []
                    for value in df_rows[attribute].values:
                        new_value.append(timestamp_transorform(value))
                    df_rows_tmp[attribute] = new_value

        processed_data[table_name] = df_rows_tmp

    return origin_data, processed_data, schema

def update_pg():
    pass

def drifts_detection():
    pass

def model_evaluate():
    pass

def e2e_update(data_folder, model_path, n_dim_dist=2, bin_size=200, bucket_method="greedy", split_date="2014-01-01 00:00:00", seed=0):
    np.random.seed(seed)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    origin_data, processed_data, schema=read_origin_data(data_folder)
    print("************************************************************")
    print(f"Training the model with data before {split_date}")
    start_time = time.time()
    train_one_stats("stats", data_folder, model_path, n_dim_dist, bin_size, bucket_method, True, actual_data=processed_data)
    print(f"training completed, took {time.time() - start_time} sec")
    
    after_data, after_data_processed = get_update_data_by_date(data_folder, processed_data=processed_data, time_date=split_date)

    # loading the trained model and buckets
    with open(os.path.join(model_path, "buckets.pkl"), "rb") as f:
        buckets = pickle.load(f)
    with open(os.path.join(model_path, f"model_stats_{bucket_method}_{bin_size}.pkl"), "rb") as f:
        FJmodel = pickle.load(f)

    update_times=5
    chunk_size=dict()
    for table in FJmodel.schema.tables:
        t_name = table.table_name
        df_rows_table = after_data[t_name]
        if df_rows_table is not None:
            chunk_size[t_name]=len(df_rows_table) // update_times
        else:
            chunk_size[t_name]=0

    for i in range(update_times):
        update_data=dict()
        for table in FJmodel.schema.tables:
            t_name = table.table_name
            ch_size=chunk_size[t_name]
            update_filename=f"{t_name}_delta{i+1}.csv"
            delta_path=os.path.join(data_folder, "update", update_filename)
            if i < update_times-1:
                left=i*ch_size
                right=(i+1)*ch_size
                data_to_db=after_data[t_name][left : right]
                data_to_model=after_data_processed[t_name][left : right]
            else:
                data_to_db=after_data[t_name][i*ch_size :]
                data_to_model=after_data_processed[t_name][i*ch_size :]
            update_data[t_name]=data_to_model
            data_to_db.to_csv(delta_path, index=False)

        update_pg()
        drifts_detection()

        start_time = time.time()
        table_buckets = FJmodel.table_buckets
        null_values = FJmodel.null_value
        data, table_buckets, null_values = update_stats_data_online(schema, model_path, buckets, table_buckets,
                                                            null_values, False, update_data)
        for table in FJmodel.schema.tables:
            t_name = table.table_name
            if t_name in data and data[t_name] is not None:
                bn = FJmodel.bns[t_name]
                bn.null_values = null_values[t_name]
                bn.update_from_data(data[t_name])
        print(f"{i+1}th update completed, took {time.time() - start_time} sec")
        model_path = os.path.join(model_path, f"updated_model_stats_{bucket_method}_{bin_size}.pkl")
        pickle.dump(FJmodel, open(model_path, 'wb'), pickle.HIGHEST_PROTOCOL)
        print(f"updated models save at {model_path}")

        model_evaluate()


if __name__ == "__main__":
    data_folder="/home/kangping/code/FactorJoin/datasets/stats_simplified_origin"
    model_path="/home/kangping/code/FactorJoin/checkpoints/update/"
    e2e_update(data_folder, model_path)