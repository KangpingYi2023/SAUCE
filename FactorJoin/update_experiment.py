import numpy as np
import pickle
import time
import os
import psycopg2
import copy
import pandas as pd
import contextlib

import sys
sys.path.append("./")
from Schemas.stats.schema import gen_stats_light_schema
from Evaluation.training import train_one_stats, test_trained_BN_on_stats
from Evaluation.testing import test_on_stats, test_on_imdb
from Join_scheme.data_prepare import update_stats_data_online
from BayesCard.Models.Bayescard_BN import Bayescard_BN
from Evaluation.updating import timestamp_transorform
from send_query import send_query


def read_table_csv(table_obj, csv_seperator=',', db_name="stats"):
    """
    Reads csv from path, renames columns and drops unnecessary columns
    """
    if db_name == "stats":
        raw_df_rows = pd.read_csv(table_obj.csv_file_location)
    elif db_name == "ssb":
        raw_df_rows = pd.read_csv(table_obj.csv_file_location, header=None, escapechar='\\', sep="|",
                              encoding="ISO-8859-1", quotechar='"')
    else:
        raw_df_rows = pd.read_csv(table_obj.csv_file_location, header=None, escapechar='\\', encoding='utf-8',
                              quotechar='"',
                              sep=csv_seperator)
    df_rows=copy.deepcopy(raw_df_rows)
    df_rows.columns = [table_obj.table_name + '.' + attr for attr in table_obj.attributes]

    for attribute in table_obj.irrelevant_attributes:
        df_rows = df_rows.drop(table_obj.table_name + '.' + attribute, axis=1)

    return df_rows.apply(pd.to_numeric, errors="ignore"), raw_df_rows.apply(pd.to_numeric, errors="ignore")


def get_update_data_by_date(schema, origin_data, processed_data, time_date="2014-01-01 00:00:00"):
    time_value = timestamp_transorform(time_date)

    after_data = dict()
    after_data_processed = dict()
    for table_obj in schema.tables:
        table_name = table_obj.table_name
        df_rows = origin_data[table_name]
        df_rows_choice = processed_data[table_name]

        idx = len(df_rows_choice)
        for attribute in df_rows_choice.columns:
            if "Date" in attribute:
                idx = np.searchsorted(df_rows_choice[attribute].values, time_value)
                break
        
        for attribute in df_rows.columns:
            if df_rows[attribute].values.dtype == "float64":
                df_rows[attribute] = df_rows[attribute].astype("Int64")
                # print(f"{table_name}.{attribute}: {df_rows[attribute].values.dtype}")

        after_data[table_name] = df_rows[idx:] if idx < len(df_rows) else None
        after_data_processed[table_name] = df_rows_choice[idx:] if idx < len(df_rows) else None
    return after_data, after_data_processed


def read_origin_data(data_path):
    if not data_path.endswith(".csv"):
        data_path += "/{}.csv"
    schema = gen_stats_light_schema(data_path)
    origin_data = dict()
    processed_data=dict()
    for table_obj in schema.tables:
        table_name = table_obj.table_name
        print(f"Loading table {table_name}")
        df_rows, raw_df_rows = read_table_csv(table_obj, db_name="stats")
        
        origin_data[table_name]=raw_df_rows

        for attribute in df_rows.columns:
            if "Date" in attribute:
                if df_rows[attribute].values.dtype == 'object':
                    new_value = []
                    for value in df_rows[attribute].values:
                        new_value.append(timestamp_transorform(value))
                    df_rows[attribute] = new_value

        processed_data[table_name] = df_rows

    return origin_data, processed_data, schema


def init_pg(dataset):
    conn_params = {
        'dbname': dataset,
        'user': 'kangping',
        'password': 'kangping',
        'host': 'localhost',
        'port': '5432'
    }

    try:
        conn = psycopg2.connect(**conn_params)
        conn.autocommit = False 
    except Exception as e:
        print(f"error: {e}")
        conn.rollback()

    return conn


def update_pg(table, delta_path, conn):
    cursor=conn.cursor()
    table=table.lower()
    try:
        with open(delta_path, "r") as delta_file:
            cursor.copy_expert(f"COPY {table} FROM STDIN WITH CSV HEADER", delta_file)
    except Exception as e:
        print(f"error: {e}")
        conn.rollback()


def drifts_detection(tables, update_data, raw_data, pool_path="./datasets/stats_simplified_origin/pool/pool_data.pkl"):
    def kl_divergence(mu1, sigma1, mu2, sigma2):
        kl = np.log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5
        # print(kl)
        return kl

    def is_drift(raw_data, sampled_data, threshold=1):
        if sampled_data is None:
            return False
        
        full_data=np.vstack((raw_data, sampled_data))

        old_mean = np.nanmean(raw_data, axis=0)
        new_mean = np.nanmean(full_data, axis=0)
        old_std = np.nanstd(raw_data, axis=0)
        new_std = np.nanstd(full_data,axis=0)
        kl_score=kl_divergence(old_mean, old_std, new_mean, new_std)
        kl_score=np.mean(kl_score)
        print(f"Distance score: {kl_score}")

        return kl_score > threshold 
    
    pool_all_tables=None
    if os.path.exists(pool_path):
        pool_all_tables=pickle.load(open(pool_path, "rb"))

    after_update_data_all=copy.deepcopy(raw_data)
    model_update_data_all=dict()
    new_pool_data_all=dict()
    need_update=False
    for table in tables:
        t_name=table.table_name
        raw_data_array=raw_data[t_name].values
        if update_data[t_name] is not None:
            update_data_array=update_data[t_name].values
        else:
            update_data_array=None

        pool_data=None
        if pool_all_tables is not None:
            pool_data = pool_all_tables[t_name]

        if pool_data is not None:
            new_data_array=np.vstack((update_data_array, pool_data.values))
        else:
            new_data_array=update_data_array
        
        if is_drift(raw_data_array, new_data_array, threshold=1e-5):
            need_update=True
            model_update_data=pd.DataFrame(new_data_array, columns=update_data[t_name].columns)
            model_update_data_all[t_name]=model_update_data
            new_pool_data_all[t_name]=None

            after_update_data_array=np.vstack((raw_data_array, new_data_array))
            after_update_data=pd.DataFrame(after_update_data_array, columns=raw_data[t_name].columns)
            after_update_data_all[t_name]=after_update_data
        else:
            model_update_data_all[t_name]=None
            if new_data_array is not None:
                pool_data_new=pd.DataFrame(new_data_array, columns=update_data[t_name].columns)
                new_pool_data_all[t_name]=pool_data_new
            else:
                new_pool_data_all[t_name]=None

    pickle.dump(pool_all_tables, open(pool_path, "wb"), pickle.HIGHEST_PROTOCOL)

    return need_update, model_update_data_all, after_update_data_all


def model_evaluate(model_path, sub_query_file, query_file, save_folder, update_type, iter):
    result_path=save_folder+"results/stats_sub_queries.txt"
    test_on_stats(model_path=model_path, query_file=sub_query_file, save_res=result_path)

    related_path="../../"
    latency_path=save_folder + "latency/"
    related_result_path=related_path+result_path
    send_query(dataset="stats", method_name=related_result_path, query_file=query_file, save_folder=latency_path, update_type=update_type, iteration=iter)
    # send_query(dataset="stats", method_name=related_result_path, query_file=query_file, save_folder=latency_path, test_type="naive", iteration=iter)


def update_model(FJmodel, model_path, model_update_data):
    table_buckets = FJmodel.table_buckets
    null_values = FJmodel.null_value
    schema = FJmodel.schema
    with open(os.path.join(model_path, "buckets.pkl"), "rb") as f:
        buckets = pickle.load(f)
    data, table_buckets, null_values = update_stats_data_online(schema, model_path, buckets, table_buckets,
                                                        null_values, False, model_update_data)
    FJmodel.table_buckets = table_buckets
    FJmodel.null_value = null_values

    for table in FJmodel.schema.tables:
        t_name = table.table_name
        if t_name in data and data[t_name] is not None:
            bn = FJmodel.bns[t_name]
            bn.null_values = null_values[t_name]
            bn.update_from_data(data[t_name])
    pickle.dump(FJmodel, open(model_path, 'wb'), pickle.HIGHEST_PROTOCOL)
    # print(f"updated models save at {model_path}")


def e2e_update(data_folder, model_path, save_folder, sub_query_file, query_file, update_type="sauce", n_dim_dist=2, bin_size=200, bucket_method="greedy", split_date="2014-01-01 00:00:00", seed=0):
    np.random.seed(seed)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    origin_data, processed_data, schema=read_origin_data(data_folder)
    print("************************************************************")
    print(f"Training the model with data before {split_date}")
    start_time = time.time()
    with open("./log.txt", "w") as log:
        with contextlib.redirect_stdout(log):
            train_one_stats("stats", data_folder, model_path, n_dim_dist, bin_size, bucket_method, True, actual_data=processed_data)
    print(f"training completed, took {time.time() - start_time} sec")
    
    after_data, after_data_processed = get_update_data_by_date(schema, origin_data, processed_data=processed_data, time_date=split_date)

    # loading the trained model and buckets
    with open(os.path.join(model_path, f"model_stats_{bucket_method}_{bin_size}.pkl"), "rb") as f:
        FJmodel = pickle.load(f)
    model_path = os.path.join(model_path, f"updated_model_stats_{bucket_method}_{bin_size}.pkl")

    update_times=5
    chunk_size=dict()
    for table in FJmodel.schema.tables:
        t_name = table.table_name
        df_rows_table = after_data[t_name]
        if df_rows_table is not None:
            chunk_size[t_name]=len(df_rows_table) // update_times
        else:
            chunk_size[t_name]=0

    db_conn=init_pg("stats")
    latency_all=[]
    for i in range(update_times):
        update_data=dict()
        for table in FJmodel.schema.tables:
            t_name = table.table_name
            ch_size=chunk_size[t_name]
            update_filename=f"{t_name}_delta{i+1}.csv"
            delta_path=os.path.join(data_folder, "update", update_filename)
            if after_data[t_name] is not None:
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
                update_pg(t_name, delta_path, db_conn)
            else:
                update_data[t_name]=None
        
        if update_type == "none":
            pass
        elif update_type == "always":
            start_time = time.time()
            update_model(FJmodel, model_path, model_update_data)
            latency = time.time() - start_time
            print(f"Update after {iter}th insertion completed, took {latency} sec")
            latency_all.append(latency)
        elif update_type == "sauce":
            need_update, model_update_data, processed_data = drifts_detection(FJmodel.schema.tables, update_data, processed_data)
            if need_update:
                start_time = time.time()
                update_model(FJmodel, model_path, model_update_data)
                latency = time.time() - start_time
                print(f"Update after {iter}th insertion completed, took {latency} sec")
                latency_all.append(latency)

        model_evaluate(model_path, sub_query_file, query_file, save_folder, update_type, i+1)
    
    print(f"Total update latency: {np.sum(latency_all)} s")


if __name__ == "__main__":
    data_folder="./datasets/stats_simplified_origin"
    model_path="./checkpoints/update/"
    save_folder="pg/"
    sub_query_file="./workloads/stats_CEB/sub_plan_queries/stats_CEB_sub_queries_10.sql"
    query_file="./workloads/stats_CEB/stats_small.sql"
    update_type="none"
    e2e_update(data_folder, model_path, save_folder, sub_query_file, query_file, update_type)