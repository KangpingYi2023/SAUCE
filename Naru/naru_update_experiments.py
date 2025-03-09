import numpy as np
import pickle
import time
import os
import psycopg2
import sys
import copy
import pandas as pd
import contextlib

sys.path.append("./")
from FactorJoin.Schemas.stats.schema import gen_stats_light_schema


def init_pg(dataset, schema):
    conn = psycopg2.connect(database=dataset, user="kangping", password="kangping", host="localhost", port=5432)
    cursor = conn.cursor()

    try:
        for table_obj in schema.tables:
            table_name = table_obj.table_name
            
            #delete all data and replace with data file
            cursor.execute(f"TRUNCATE TABLE {table_name};")
            with open(table_obj.csv_file_location, "r") as csv_file:
                cursor.copy_expert(f"COPY {table_name} FROM STDIN WITH CSV HEADER", csv_file)
            print(f"Table {table_name} init finished!")

        conn.commit()

    except Exception as e:
        print(f"error: {e}")
        conn.rollback()

    cursor.close()
    conn.close()


def read_data_and_init_db(dataset, src_data_path):
    assert os.path.isdir(src_data_path), "src_data_path need to be a directory!"

    end2end_data_path = os.path.join(src_data_path, "end2end")
    if not os.path.exists(end2end_data_path):
        os.mkdir(end2end_data_path)

    if dataset in ["stats", "imdb"]:
        table_file_list = os.listdir(src_data_path)
        
    else:
        raise ValueError(f"Unknown dataset {dataset}!")
    
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
    
    init_pg(dataset, schema)

    return origin_data, processed_data, schema


def e2e_update(dataset, data_folder, model_folder, pg_folder, sub_query_file, query_file, update_type="sauce", update_times=5, n_dim_dist=2, bin_size=200, bucket_method="greedy", split_date="2014-01-01 00:00:00", seed=0):
    np.random.seed(seed)
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    origin_data, processed_data, schema=read_data_and_init_db(dataset, data_folder)
    
    
    after_data, after_data_processed = get_update_data_by_date(schema, origin_data, processed_data=processed_data, time_date=split_date)

    # loading the trained model and buckets
    model_path=os.path.join(model_folder, f"model_stats_{bucket_method}_{bin_size}.pkl")
    with open(model_path, "rb") as f:
        FJmodel = pickle.load(f)

    chunk_size=dict()
    for table in FJmodel.schema.tables:
        t_name = table.table_name
        df_rows_table = after_data[t_name]
        if df_rows_table is not None:
            chunk_size[t_name]=len(df_rows_table) // update_times
        else:
            chunk_size[t_name]=0

    latency_all=[]
    for i in range(update_times):
        update_data=dict()
        for table in FJmodel.schema.tables:
            t_name = table.table_name
            ch_size=chunk_size[t_name]
            update_filename=f"{t_name}_delta{i+1}.csv"
            delta_path=os.path.join(pg_folder, "update", update_filename)
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
                update_pg(dataset, t_name, delta_path)
            else:
                update_data[t_name]=None
        
        if update_type == "none":
            pass
        elif update_type == "naive":
            pass
        elif update_type == "always":
            start_time = time.time()
            update_model(dataset, FJmodel, model_folder, model_path, update_data)
            latency = time.time() - start_time
            print(f"Update after {i+1}th insertion completed, took {latency} sec\n")
            latency_all.append(latency)
        elif update_type == "sauce":
            need_update, model_update_data, processed_data = drifts_detection(FJmodel.schema.tables, update_data, processed_data)
            if need_update:
                start_time = time.time()
                update_model(dataset, FJmodel, model_folder, model_path, model_update_data)
                latency = time.time() - start_time
                print(f"Update after {i+1}th insertion completed, took {latency} sec\n")
                latency_all.append(latency)
        else:
            raise ValueError(f"Unknown update type '{update_type}'")

        model_evaluate(dataset, model_path, sub_query_file, query_file, pg_folder, update_type, i+1)
    
    print(f"Total update latency: {np.sum(latency_all)}s\n")