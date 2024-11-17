import pandas as pd

if __name__ == "__main__":
    black_list=[120, 58, 122, 126, 135, 68, 108]
    original_query_path="./stats_CEB/stats_CEB.sql"
    original_sub_query_path="./stats_CEB/sub_plan_queries/stats_CEB_sub_queries.sql"
    original_sub_query_results_path="./stats_CEB/sub_plan_queries/stats_CEB_sub_queries_results.csv"

    target_query_path="./stats_CEB/stats_light.sql"
    target_sub_query_path="./stats_CEB/sub_plan_queries/stats_CEB_sub_queries_light.sql"
    target_sub_query_results_path="./stats_CEB/sub_plan_queries/stats_CEB_sub_queries_light_results.csv"

    with open(original_query_path, "r") as oqf:
        raw_queries=oqf.readlines()

    with open(original_sub_query_path, "r") as osqf:
        raw_sub_queries=osqf.readlines()

    raw_sub_queries_results=pd.read_csv(original_sub_query_results_path, header=None)
    raw_sub_queries_results=raw_sub_queries_results.to_numpy()

    with open(target_query_path, "w") as tqf:
        for i, query_str in enumerate(raw_queries):
            if i+1 in black_list:
                continue
            else:
                tqf.write(query_str)
    
    with open(target_sub_query_path, "w") as tsqf:
        with open(target_sub_query_results_path, "w") as tsqrf:
            for i, sub_query_str in enumerate(raw_sub_queries):
                try:
                    number=int(sub_query_str.split("||")[-1])
                except:
                    print(f"{number} not int!")
                
                if number in black_list:
                    continue
                else:
                    tsqf.write(sub_query_str)
                    tsqrf.write(f"{int(raw_sub_queries_results[i])}\n")
