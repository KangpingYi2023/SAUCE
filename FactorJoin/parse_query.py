import pickle
import time
import numpy as np
import ast

OPS = {
    ">": np.greater,
    "<": np.less,
    ">=": np.greater_equal,
    "<=": np.less_equal,
    "=": np.equal,
}

STATS={
	"short_name_to_full": {
		"u": "users",
		"b": "badges",
		"c": "comments",
		"ph": "postHistory",
		"v": "votes",
		"pl": "postLinks",
		"t": "tags",
    }
}

def timestamp_transorform(time_string, start_date="2010-07-19 00:00:00"):
    start_date_int = time.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    time_array = time.strptime(time_string, "%Y-%m-%d %H:%M:%S")
    return int(time.mktime(time_array)) - int(time.mktime(start_date_int))

def process_condition(cond, tables_all=None):
    # parse a condition, either filter predicate or join operation
    start = None
    join = False
    join_keys = {}

    for i in range(len(cond)):
        s = cond[i]
        if s in OPS:
            start = i
            if cond[i + 1] in OPS:
                end = i + 2
            else:
                end = i + 1
            break
    assert start is not None
    left = cond[:start].strip()
    ops = cond[start:end].strip()
    right = cond[end:].strip()
    table1 = left.split(".")[0].strip().lower()
    if tables_all:
        cond = cond.replace(table1 + ".", tables_all[table1] + ".")
        table1 = tables_all[table1]
        left = table1 + "." + left.split(".")[-1].strip()
    if "." in right:
        table2 = right.split(".")[0].strip().lower()
        if tables_all:
            cond = cond.replace(table2 + ".", tables_all[table2] + ".")
            table2 = tables_all[table2]
            right = table2 + "." + right.split(".")[-1].strip()
        join = True
        join_keys[table1] = left
        join_keys[table2] = right
        return table1 + " " + table2, cond, join, join_keys

    value = right
    try:
        value = list(ast.literal_eval(value.strip()))
    except:
        try:
            value = int(value)
        except:
            try:
                value = float(value)
            except:
                value = value

    return table1, [left, ops, value], join, join_keys

def parse_query(query_str):
    tables_all=dict()
    table_predicates=dict()
    join_cond=dict()
    true_card = int(query_str.split("||")[-1])

    query = query_str.split("||")[0][:-1]
    t = time.time()
    predicates_str=query.split(" WHERE ")[-1].split(" AND ")
    tables_str=query.split(" WHERE ")[0].split(" FROM ")[-1].split(",")
    
    for table_str in tables_str:
        table_str = table_str.strip()
        if " as " in table_str:
            tables_all[table_str.split(" as ")[-1]] = table_str.split(" as ")[0]
        else:
            tables_all[table_str.split(" ")[-1]] = table_str.split(" ")[0]

    for tab in tables_all.values():
        table_predicates[tab] = dict()
        table_predicates[tab]["cols"] = []
        table_predicates[tab]["ops"] = []
        table_predicates[tab]["vals"] = []
        table_predicates[tab]["join_keys"] = set()

    for predicate_str in predicates_str:
        table, cond, join, join_key = process_condition(predicate_str, tables_all)

        if not join:
            attr = cond[0]
            op = cond[1]
            value = cond[2]
            if "Date" in attr:
                assert "::timestamp" in value  # this is hardcoded for STATS-CEB workload
                value = timestamp_transorform(value.split("::timestamp")[0].strip("'"))

            table_predicates[table]["cols"].append(attr.split(".")[-1])
            table_predicates[table]["ops"].append(op)
            table_predicates[table]["vals"].append(value)
            
        else:
            table1=table.split(" ")[0]
            table2=table.split(" ")[-1]
            join_cond[cond]=(table1, table2)
            join_key1=join_key[table1].split(".")[-1]
            join_key2=join_key[table2].split(".")[-1]
            table_predicates[table1]["join_keys"].add(join_key1)
            table_predicates[table2]["join_keys"].add(join_key2)

    return tables_all, table_predicates, join_cond, true_card

def read_queries_on_stats(query_file, save_res=None):
    with open(query_file, "r") as f:
	    queries = f.readlines()

    log_path="./Naru/checkpoints/query_parse_log.txt"
    with open(log_path, "w") as log_file:
        for i, query_str in enumerate(queries):
            tables_all, table_predicates, join_cond, true_card = parse_query(query_str)
            if i%10==0:
                log_file.write(f"{tables_all}\n")
                for tab in table_predicates:
                    log_file.write(f"{tab}:{table_predicates[tab]}\n")
                log_file.write(f"{join_cond.keys()}\n")
                log_file.write(f"true card: {true_card}\n")

    print("Query parse finished!")
        	

if __name__ == "__main__":
	query_path="/home/kangping/code/End-to-End-CardEst-Benchmark/workloads/stats_CEB/sub_plan_queries/stats_CEB_sub_queries.sql"
	read_queries_on_stats(query_path)
