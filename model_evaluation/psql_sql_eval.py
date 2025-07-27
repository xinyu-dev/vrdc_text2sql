import argparse
import json
import os
from datetime import datetime
from typing import Union

import pandas as pd
import yaml
from eval import compare_query_results
from tqdm import tqdm
from utils import postprocess_sql_query_from_markdown, fix_parentheses_balance


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sql_eval.yaml")
    parser.add_argument("--pred_file", type=str, default="model_predictions/model_base/mistral-nemo-minitron-8b-instruct/mistral-nemo-minitron-8b-instruct_vinmec_results.json")
    parser.add_argument("--out_file", type=str, default="results")
    parser.add_argument("--input_template", type=str, default="")
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    error_log_base = config["error_log_base"]
    database_credentials = config["database_credentials"]

    if args.pred_file.endswith(".jsonl"):
        with open(args.pred_file, 'r', encoding='utf-8') as f:
            lines = [json.loads(x.strip()) for x in f.readlines()]
    else:
        with open(args.pred_file, 'r', encoding='utf-8') as f:
            lines = json.loads(f.read())

    idx=0
    # print("Example:")
    # print("\tSQL Query: \n", postprocess_sql_query_from_markdown(lines[idx]["output"]), '\n')
    # print("\tGenerated SQL Query: \n", postprocess_sql_query_from_markdown(lines[idx]["predict"]))

    # convert data from json to pandas
    processed_data = []
    for line in lines:
        predict_value = line["predict"]
        # Add ```sql in front if it doesn't already have it
        if predict_value and not predict_value.strip().startswith("```sql"):
            predict_value = f"```sql\n{predict_value}\n```"
        
        processed_data.append((line.get("input", "user_query"), line["output"], predict_value))
    df = pd.DataFrame(processed_data, columns=["input", "sql_query", "generated_sql"])

    # Create log files
    task_name = os.path.basename(args.pred_file).rsplit(".", 1)[0]
    save_rp = args.out_file
    os.makedirs(save_rp, exist_ok=True)
    save_task_rp = os.path.join(save_rp, task_name)
    os.makedirs(save_task_rp, exist_ok=True)
    save_error_rp = os.path.join(save_task_rp, "error_logs")
    os.makedirs(save_error_rp, exist_ok=True)
    save_path = os.path.join(save_task_rp, f"results.csv")
    
    # create cache, remove duplicate questions when re-run
    questions = []
    if os.path.exists(save_path):
        save_df = pd.read_csv(save_path)
        for idx, row in save_df.iterrows():
            questions.append(row["question"])
    else:
        save_df = pd.DataFrame([], columns=["question", "sql_query", "generated_sql_query", "status", "results_gold", "results_gen"])
    df = df[~df["input"].isin(questions)]

    # evaluation loop
    save_metrics_path = os.path.join(save_task_rp, f"metrics.json")
    if os.path.exists(save_metrics_path):
        with open(save_metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
            exactly_same, subset, total, predict_run_errors = metrics["exactly_same"], metrics["subset"], metrics["total (run successfully)"], metrics["predict_run_errors"]
    else:
        exactly_same, subset, total, predict_run_errors = 0, 0, 0, 0

    pbar = tqdm(
        df.iterrows(),
        total=len(df.index),
        desc=f"{task_name} evaluation",
    )
    for idx, row in pbar:
        pbar.set_postfix({'total': total, 'exactly_same': exactly_same, 'subset': subset, 'predict_run_errors': predict_run_errors})
        
        if args.input_template == "chat":
            messages = row['input']
            if type(messages) == str:
                messages = json.loads(messages)
            question = messages[-1]["content"]
        else:
            question = row['input']
        
        if row["generated_sql"] is None:
            predict_run_errors += 1
            continue
        
        sql_query = fix_parentheses_balance(postprocess_sql_query_from_markdown(row["sql_query"]))
        generated_sql_query = fix_parentheses_balance(postprocess_sql_query_from_markdown(row["generated_sql"]))
        
        if sql_query is None:
            print(f"Error: {row['sql_query']}")
            continue
        
        try:
            result, results_gold, results_gen = compare_query_results(
                query_gold=sql_query,
                query_gen=generated_sql_query,
                db_name=database_credentials["database"],
                db_type=config["database_type"],
                db_creds=database_credentials,
                question=question,
                query_category="",
                timeout=120
            )
        except Exception as e:
            print(f"Error: {e}")
            continue
        
        if result is None:
            continue
        
        if None in result:
            predict_run_errors += 1
            with open(os.path.join(save_error_rp, f"{idx}.md"), "w") as f:
                f.write(error_log_base.format(
                    question=question,
                    sql_query=sql_query,
                    generated_sql_query=generated_sql_query
                ))
                f.write(f"\n3. ERROR\n ```commandline\n{result[1]}```")

            save_df.loc[len(save_df.index)] = {
                "question": question,
                "sql_query": sql_query,
                "generated_sql_query": generated_sql_query,
                "status": ["run_error"],
                "results_gold": results_gold,
                "results_gen": results_gen,
            }
            continue          
            
        if None in result:
            predict_run_errors += 1
            continue
        
        statuses = []
        if result[0]:
            exactly_same += 1
            statuses.append("exactly_same")

        if result[1]:
            subset += 1
            statuses.append("subset")

        if not(result[0] or result[1]):
            with open(os.path.join(save_error_rp, f"{idx}.md"), "w") as f:
                f.write(error_log_base.format(
                    question=question,
                    sql_query=sql_query,
                    generated_sql_query=generated_sql_query
                ))
            statuses.append("error")

        save_df.loc[len(save_df.index)] = {
            "question": question, 
            "sql_query": sql_query, 
            "generated_sql_query": generated_sql_query, 
            "status": statuses,
            "results_gold": results_gold,
            "results_gen": results_gen,
        }
        # except Exception as e:
        #     continue

        total += 1
    save_df.to_csv(save_path, index=False, encoding="utf-8", index_label=False)
        
    print(f"exactly same: {exactly_same}, subset: {subset}, total: {total}, predict_run_errors: {predict_run_errors}")

    exactly_acc = exactly_same / (total + predict_run_errors)
    subset_acc = subset / (total + predict_run_errors)

    print(f"Accuracy:\t exactly: {exactly_acc}, subset: {subset_acc}")
    
 
    with open(save_metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "total (run successfully)": total,
            "exactly_same": exactly_same,
            "subset": subset,
            "predict_run_errors": predict_run_errors,
            "exactly_acc": exactly_acc,
            "subset_acc": subset_acc,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=4, ensure_ascii=False)
