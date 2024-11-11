import pickle
import time
import copy
import ast
import sys
import os
import pandas as pd
import numpy as np


from parse_query import parse_query

sys.path.append("../")
from end2end.data_updater import create_sampler, DataUpdater


OPS = {
    '>': np.greater,
    '<': np.less,
    '>=': np.greater_equal,
    '<=': np.less_equal,
    '=': np.equal,
    '==': np.equal
}


MIN_MAX_Qerrors = {
    'badges':       {"mean": 2.12, "median": 2.1},
    'votes':        {"mean": 25, "median": 6},
    'postHistory':  {"mean": 12, "median": 5},
    'posts':        {"mean": 8, "median": 3.5},
    'users':        {"mean": 5, "median": 3},
    'comments':     {"mean": 7, "median": 4},
    'postLinks':    {"mean": 7, "median": 4},
    'tags':         {"mean": 2, "median": 2},
}


sample_lib=["permute", "tupleskew", "valueskew"]
fraction_range=[0.1, 0.3]
skew_ratio_lib=[1e-1, 1e-2, 1e-3]


def str_pattern_matching(s):
    # split the string "attr==value" to ('attr', '=', 'value')
    op_start = 0
    if len(s.split(' IN ')) != 1:
        s = s.split(' IN ')
        attr = s[0].strip()
        try:
            value = list(ast.literal_eval(s[1].strip()))
        except:
            temp_value = s[1].strip()[1:][:-1].split(',')
            value = []
            for v in temp_value:
                value.append(v.strip())
        return attr, 'in', value

    for i in range(len(s)):
        if s[i] in OPS:
            op_start = i
            if i + 1 < len(s) and s[i + 1] in OPS:
                op_end = i + 1
            else:
                op_end = i
            break
    attr = s[:op_start]
    value = s[(op_end + 1):].strip()
    ops = s[op_start:(op_end + 1)]
    try:
        value = list(ast.literal_eval(s[1].strip()))
    except:
        try:
            value = int(value)
        except:
            try:
                value = float(value)
            except:
                value = value
    return attr.strip(), ops.strip(), value


def construct_table_query(BN, table_query, attr, ops, val, epsilon=1e-6):
    if BN is None or attr not in BN.attr_type:
        return None
    if BN.attr_type[attr] == 'continuous':
        if ops == ">=":
            query_domain = (val, np.infty)
        elif ops == ">":
            query_domain = (val + epsilon, np.infty)
        elif ops == "<=":
            query_domain = (-np.infty, val)
        elif ops == "<":
            query_domain = (-np.infty, val - epsilon)
        elif ops == "=" or ops == "==":
            query_domain = val
        else:
            assert False, f"operation {ops} is invalid for continous domain"

        if attr not in table_query:
            table_query[attr] = query_domain
        else:
            prev_l = table_query[attr][0]
            prev_r = table_query[attr][1]
            query_domain = (max(prev_l, query_domain[0]), min(prev_r, query_domain[1]))
            table_query[attr] = query_domain

    else:
        attr_domain = BN.domain[attr]
        if type(attr_domain[0]) != str:
            attr_domain = np.asarray(attr_domain)
        if ops == "in":
            assert type(val) == list, "use list for in query"
            query_domain = val
        elif ops == "=" or ops == "==":
            if type(val) == list:
                query_domain = val
            else:
                query_domain = [val]
        else:
            if type(val) == list:
                assert len(val) == 1
                val = val[0]
                assert (type(val) == int or type(val) == float)
            operater = OPS[ops]
            query_domain = list(attr_domain[operater(attr_domain, val)])

        if attr not in table_query:
            table_query[attr] = query_domain
        else:
            query_domain = [i for i in query_domain if i in table_query[attr]]
            table_query[attr] = query_domain

    return table_query


def timestamp_transorform(time_string, start_date="2010-07-19 00:00:00"):
    start_date_int = time.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    time_array = time.strptime(time_string, "'%Y-%m-%d %H:%M:%S'")
    return int(time.mktime(time_array)) - int(time.mktime(start_date_int))


def parse_query_single_table(query, BN, table_name):
    useful = query.split(' WHERE ')[-1].strip()
    result = dict()
    if 'WHERE' not in query:
        return result
    for sub_query in useful.split(' AND '):
        attr, ops, value = str_pattern_matching(sub_query.strip())
        attr = table_name + "." + attr.split(".")[-1]
        if "Date" in attr:
            assert "::timestamp" in value
            value = timestamp_transorform(value.strip().split("::timestamp")[0])
        construct_table_query(BN, result, attr, ops, value)
    return result


def read_table_csv(table_obj, csv_seperator=',', db_name="stats"):
    """
    Reads csv from path, renames columns and drops unnecessary columns
    """
    if db_name == "stats":
        raw_df_rows = pd.read_csv(table_obj.csv_file_location)
    else:
        raw_df_rows = pd.read_csv(table_obj.csv_file_location, header=None, escapechar='\\', encoding='utf-8',
                              quotechar='"',
                              sep=csv_seperator)
    df_rows=copy.deepcopy(raw_df_rows)
    df_rows.columns = [table_obj.table_name + '.' + attr for attr in table_obj.attributes]

    for attribute in table_obj.irrelevant_attributes:
        df_rows = df_rows.drop(table_obj.table_name + '.' + attribute, axis=1)

    return df_rows.apply(pd.to_numeric, errors="ignore")


def get_groundtruth(query, table_obj, raw_data):
    if "WHERE" not in query:
        return table_obj.table_size
    else:
        conditions=query.split("WHERE")[-1].split(" AND ")
        bools=None
        for cond in conditions:
            attr, ops, value = str_pattern_matching(cond.strip())
            attr = table_obj.table_name + "." + attr.split(".")[-1]
            if "Date" in attr:
                assert "::timestamp" in value
                value = timestamp_transorform(value.strip().split("::timestamp")[0])
            
            # row_data=df_rows[attr].values
            filter=OPS[ops](raw_data, value)
            if bools is None:
                bools = filter
            else:
                bools *= filter
        cardinality = bools.sum()

    return cardinality


def kl_divergence(mu1, sigma1, mu2, sigma2):
    kl = np.log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5
    # print(kl)
    return kl


def distance_compute(raw_data, updated_data):   
    old_mean = np.nanmean(raw_data, axis=0)
    new_mean = np.nanmean(updated_data, axis=0)
    old_std = np.nanstd(raw_data, axis=0)
    new_std = np.nanstd(updated_data,axis=0)
    kl_score=kl_divergence(old_mean, old_std, new_mean, new_std)
    kl_score=np.mean(kl_score)
    # print(f"Distance score: {kl_score}")

    return kl_score


def divide_workloads_by_table(queries_str):
    workloads=dict()
    for i, query_str in enumerate(queries_str):
        true_card = int(query_str.split("||")[-1])
        query = query_str.split("||")[0][:-1]
        # print(query)
        tables_str=query.split(" WHERE ")[0].split(" FROM ")[-1].split(",")

        tables_all=[]
        for table_str in tables_str:
            table_str = table_str.strip()
            if " as " in table_str:
                tables_all.append(table_str.split(" as ")[0])
            else:
                tables_all.append(table_str.split(" ")[0])
        
        assert len(tables_all)==1, f"Error, query {i+1} includes more than 1 table!"
        table=tables_all[0]
        
        if table not in workloads:
            workloads[table]=[query]
        else:
            workloads[table].append(query)

    return workloads


def bootstrap_single_table(table_obj, raw_data, bn, workload, output_path):
    t_name=table_obj.table_name
    distance_all=[]
    qerror_median_all=[]
    qerror_mean_all=[]
    for i in range(bootstrap):
        sampler_type=np.random.choice(sample_lib)
        update_fraction=np.random.uniform(fraction_range[0], fraction_range[1])
        skew_ratio=np.random.choice(skew_ratio_lib)

        sampler=create_sampler(sampler_type=sampler_type, update_fraction=update_fraction, skew_ratio=skew_ratio)
        data_updater=DataUpdater(raw_data, sampler)
        data_updater.update_data()
        updated_data=data_updater.get_updated_data()       

        distance=distance_compute(raw_data, updated_data)
        distance_all.append(distance)

        qerrors=[]
        for query in workload:
            table_query = parse_query_single_table(query, bn, t_name)
            prob, __ = bn.query(table_query, return_prob=True)
            pred = prob * updated_data.size
            if isinstance(pred, np.ndarray):
                pred = pred.item()
            true_card = get_groundtruth(query, table_obj, updated_data)

            if pred == 0:
                pred=1
            if true_card == 0:
                true_card=1

            qerror = max(pred / true_card, true_card / pred)
            qerrors.append(qerror)

        qerror_median=np.percentile(qerrors, 50)
        qerror_mean=np.mean(qerrors)
        qerror_median_all.append(qerror_median)
        qerror_mean_all.append(qerror_mean)

        # print(f"Table {t_name}: {len(workloads[t_name])} queries")
        # print(f"q-error mean is {qerror_mean}")
        # print(f"q-error 50% percentile is {qerror_median}")
        if (i+1)%1000 == 0:
            print(f"{i+1}th bootstrap on table {t_name} finished!")

    df=pd.DataFrame({
        "distances": distance_all,
        "qerror_medians": qerror_median_all,
        "qerror_means": qerror_mean_all
    })

    df.to_csv(output_path, index=False)


def analyze_threshold(table, save_path, FP=1e-2, min_max_mean_qerror=3, min_max_median_qerror=3):
    df_raw=pd.read_csv(save_path)

    df_filtered_by_median=df_raw[df_raw["qerror_medians"]>min_max_median_qerror]
    df_filtered_by_mean=df_raw[df_raw["qerror_means"]>min_max_mean_qerror]

    assert not df_filtered_by_median.empty, f"No median q-error in {table} greater than {min_max_median_qerror}"
    assert not df_filtered_by_mean.empty, f"No mean q-error in {table} greater than {min_max_mean_qerror}"
    threshold_by_median=np.quantile(df_filtered_by_median["distances"].values, FP)
    threshold_by_mean=np.quantile(df_filtered_by_mean["distances"].values, FP)

    return threshold_by_median, threshold_by_mean


def analyze_all_tables(bound_ensemble, workloads):
    thres_by_mean_all=dict()
    thres_by_median_all=dict()
    for table_obj in bound_ensemble.schema.tables:
        t_name=table_obj.table_name
        bn=bound_ensemble.bns[t_name]
        df_rows=read_table_csv(table_obj, db_name="stats")
        raw_data=df_rows.values

        output_path=os.path.join(output_folder, f"{t_name}_threshold_boostrap.csv")
        # bootstrap_single_table(table_obj, raw_data, bn, workloads[t_name], output_path)
        mmq_mean=MIN_MAX_Qerrors[t_name]["mean"]
        mmq_median=MIN_MAX_Qerrors[t_name]["median"]
        thres_by_median, thres_by_mean = analyze_threshold(t_name, output_path, FP=5e-2, min_max_mean_qerror=mmq_mean, min_max_median_qerror=mmq_median)

        thres_by_median_all[t_name] = thres_by_median
        thres_by_mean_all[t_name] = thres_by_mean

    return thres_by_mean_all, thres_by_median_all


if __name__ == "__main__":
    data_folder="./datasets/stats_simplified/"
    model_path="./checkpoints/model_stats_greedy_200.pkl"
    query_path="./workloads/stats_CEB/sub_plan_queries/stats_CEB_single_table_sub_query.sql"
    output_folder="./checkpoints/bootstrap"
    bootstrap=1000

    with open(model_path, "rb") as mf:
        bound_ensemble = pickle.load(mf)

    with open(query_path, "r") as qf:
        queries_str = qf.readlines()
    # print(len(queries_str))

    workloads=divide_workloads_by_table(queries_str)

    thres_by_mean_all, thres_by_median_all = analyze_all_tables(bound_ensemble, workloads)
    thres_path=os.path.join(output_folder, "stats_thresholds.txt")
    with open(thres_path, "w") as output_file:
        for table in thres_by_median_all:
            thres_median=thres_by_median_all[table]
            thres_mean=thres_by_mean_all[table]

            output_file.write(f"Stats-{table} recommend threshold by mean {thres_mean}, by median {thres_median}\n")