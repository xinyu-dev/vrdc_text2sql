'''Official evaluation script for EHRSQL'''

import argparse
import json
import multiprocessing as mp
import os
import re
import sqlite3
import sys
from collections import OrderedDict
from datetime import datetime

import numpy as np
from func_timeout import FunctionTimedOut, func_timeout
from utils import postprocess_sql_query_from_markdown


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--pred_file', metavar='pred.json', help='model predictions')
    args.add_argument('--db_path', required=True, type=str, help='path database')
    args.add_argument("--num_workers", type=int, default=-1)
    args.add_argument("--timeout", type=int, default=60.0, help='execution time limit in sec')
    args.add_argument("--out_file", type=str, default=None, help='path to save the output file')
    args.add_argument("--ndigits", type=int, default=2, help='scores rounded to ndigits')
    args.add_argument("--current_time", type=str, default="2105-12-31 23:59:00")
    return args.parse_args()

def post_process_sql(query,
                     current_time="2105-12-31 23:59:00",
                     precomputed_dict={
                                'temperature': (35.5, 38.1),
                                'sao2': (95.0, 100.0),
                                'heart rate': (60.0, 100.0),
                                'respiration': (12.0, 18.0),
                                'systolic bp': (90.0, 120.0),
                                'diastolic bp':(60.0, 90.0),
                                'mean bp': (60.0, 110.0)
                            }):
    query = query.lower()
    if "current_time" in query:
        query = query.replace("current_time", f"'{current_time}'")
    if re.search('[ \n]+([a-zA-Z0-9_]+_lower)', query) and re.search('[ \n]+([a-zA-Z0-9_]+_upper)', query):
        vital_lower_expr = re.findall('[ \n]+([a-zA-Z0-9_]+_lower)', query)[0]
        vital_upper_expr = re.findall('[ \n]+([a-zA-Z0-9_]+_upper)', query)[0]
        vital_name_list = list(set(re.findall('([a-zA-Z0-9_]+)_lower', vital_lower_expr) + re.findall('([a-zA-Z0-9_]+)_upper', vital_upper_expr)))
        if len(vital_name_list)==1:
            processed_vital_name = vital_name_list[0].replace('_', ' ')
            if processed_vital_name in precomputed_dict:
                vital_range = precomputed_dict[processed_vital_name]
                query = query.replace(vital_lower_expr, f"{vital_range[0]}").replace(vital_upper_expr, f"{vital_range[1]}")
    query = query.replace("''", "'").replace('< =', '<=')
    query = query.replace("%y", "%Y").replace('%j', '%J')
    query = query.replace("'now'", f"'{current_time}'")
    query = query.replace("today", f"'{current_time}'")
    query = query.replace("current_timestamp", f"'{current_time}'")
    query = query.replace("strftime('%J')", f"strftime('%J', '{current_time}')")
    query = query.replace("strftime('%Y')", f"strftime('%Y', '{current_time}')")
    query = query.replace("strftime('%m')", f"strftime('%m', '{current_time}')")
    query = query.replace("strftime('%d')", f"strftime('%d', '{current_time}')")
    query = query.replace("strftime('%H')", f"strftime('%H', '{current_time}')")
    query = query.replace("strftime('%M')", f"strftime('%M', '{current_time}')")
    query = query.replace("strftime('%S')", f"strftime('%S', '{current_time}')")
    return query

def process_answer(ans):
    """
    This function standardizes the result by taking up to the first 100 rows, converting each row to a string, sorting them alphabetically, and finally, creating a single string representation of the sorted list.
The evaluation is then a simple string comparison between the processed result of the ground-truth query and the processed result of the predicted query.
    """
    return str(sorted([str(ret) for ret in ans[:100]])) # check only up to 100th record

def execute(sql, db_path):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    result = cur.execute(sql).fetchall()
    con.close()
    return result

def execute_wrapper(sql, args, tag, skip_indicator='null'):
    if sql != skip_indicator:
        try:
            result = func_timeout(args.timeout, execute, args=(sql, args.db_path))
        except KeyboardInterrupt:
            sys.exit(0)
        except FunctionTimedOut:
            result = [(f'timeout_{tag}',)]
        except:
            result = [(f'error_{tag}',)] # possibly len(query) > 512 or not executable
        result = process_answer(result)
    else:
        result = skip_indicator
    return result

def execute_query(sql1, sql2, args, data_idx=None):
    '''
    Execute the query. Time out if it exceeds {args.timeout} seconds
    '''
    result1 = execute_wrapper(sql1, args, tag='real')
    result2 = execute_wrapper(sql2, args, tag='pred')
    result = {'data_idx': data_idx, 'real': result1, 'pred': result2}
    return result

def execute_query_distributed(real, pred, db_path, num_workers):

    exec_result = []
    def result_tracker(result):
        exec_result.append(result)

    pool = mp.Pool(processes=num_workers)
    for data_idx, (sql1, sql2) in enumerate(zip(real, pred)):
        pool.apply_async(execute_query, args=(sql1, sql2, args, data_idx), callback = result_tracker)
    pool.close()
    pool.join()

    return exec_result


def process_input_data(input_context: str) -> str:
    question = input_context.split("<extra_id_1>User")[-1].strip()[:-len("<extra_id_1>Assistant")].strip()
    return question


def main(args):

    if not os.path.exists(args.db_path):
        raise Exception('Database does not exist: %s' % args.db_path)
    
    num_workers = mp.cpu_count() if args.num_workers==-1 else args.num_workers
    
    if args.pred_file.endswith('.jsonl'):
        with open(args.pred_file, 'r', encoding='utf-8') as f:
            pred = [json.loads(x) for x in f]
    else:
        with open(args.pred_file, 'r', encoding='utf-8') as f:
            pred = json.load(f)

    data_id, question, query_real, query_pred = [],[],[],[]
    for idx, line in enumerate(pred):
        data_id.append(idx)
        # question.append(process_input_data(line['input']))
        input_text = line.get('input') or line.get('question') or line.get('user_query')
        if input_text is None:
            raise KeyError("None of 'input', 'question', or 'user_query' found in the input line.")
        question.append(process_input_data(input_text))
        # query_real.append(postprocess_sql_query_from_markdown(post_process_sql(line['output'])))
        # if line['predict'] is None:
        # Handle output/real
        real_sql = line.get('output') or line.get('real')
        if real_sql is None:
            raise KeyError("None of 'output' or 'real' found in the input line.")
        query_real.append(postprocess_sql_query_from_markdown(post_process_sql(real_sql)))
        # Handle predict/pred
        pred_sql = line.get('predict') if 'predict' in line else line.get('pred')
        if pred_sql is None:
            query_pred.append('null')
        else:
            # query_pred.append(postprocess_sql_query_from_markdown(line['predict']))
            query_pred.append(postprocess_sql_query_from_markdown(post_process_sql(pred_sql)))

    exec_real, exec_pred = [],[]
    if num_workers>1:
        exec_result = execute_query_distributed(query_real, query_pred, args.db_path, num_workers)
        indices = []
        for ret in exec_result:
            exec_real.append(ret['real'])
            exec_pred.append(ret['pred'])
            indices.append(ret['data_idx'])
        exec_real = np.array(exec_real)[np.argsort(indices)]
        exec_pred = np.array(exec_pred)[np.argsort(indices)]
    else:
        for sql1, sql2 in zip(query_real, query_pred):
            ret = execute_query(sql1, sql2, args)
            exec_real.append(ret['real'])
            exec_pred.append(ret['pred'])
        exec_real = np.array(exec_real)
        exec_pred = np.array(exec_pred)
    
    # Calculate individual scores and save results to JSONL file
    if args.out_file:
        if os.path.isdir(args.out_file):
            # If out_file is a directory, append a filename
            output_file = os.path.join(args.out_file, os.path.basename(args.pred_file).rsplit('.', 1)[0] + '_evaluation_results.jsonl')
        else:
            output_file = args.out_file
    else:
        output_file = os.path.splitext(args.pred_file)[0] + '_results.jsonl'

    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, (line, real_result, pred_result) in enumerate(zip(pred, exec_real, exec_pred)):
            # Calculate individual sample scores
            sample_scores = {}
            
            # Check if both queries failed to execute
            both_failed = ('error_real' in real_result and 'error_pred' in pred_result)
            
            # Check if real query failed and pred returned empty or None
            real_failed_pred_empty = ('error_real' in real_result and (pred_result == '[]' or pred_result == "['(None,)']"))
            
            # Precision score for this sample
            if pred_result != 'null' and pred_result is not None:
                sample_scores['precision_ans'] = 1 if real_result not in ['null', None] else 0
                sample_scores['precision_exec'] = 1 if real_result == pred_result or both_failed or real_failed_pred_empty else 0
            
            # Recall score for this sample
            if real_result != 'null' and real_result is not None:
                sample_scores['recall_ans'] = 1 if pred_result not in ['null', None] else 0
                sample_scores['recall_exec'] = 1 if real_result == pred_result or both_failed or real_failed_pred_empty else 0
            
            # Accuracy for this sample
            sample_scores['accuracy'] = 1 if real_result == pred_result or both_failed or real_failed_pred_empty else 0
            
            # Create result dictionary with all information
            result_dict = line.copy()  # Copy original input
            result_dict['input'] = question[idx]
            result_dict['pred'] = query_pred[idx]
            result_dict['real_result'] = real_result
            result_dict['pred_result'] = pred_result
            result_dict['sample_scores'] = sample_scores
            f.write(json.dumps(result_dict) + '\n')

    precision_ans_list, precision_exec_list, recall_ans_list, recall_exec_list, acc_list, error_list = [], [], [], [], [], []
    for idx in range(len(exec_real)):
        ans_real, ans_pred = exec_real[idx], exec_pred[idx]
        
        # Check if both queries failed to execute
        both_failed = ('error_real' in ans_real and 'error_pred' in ans_pred)
        
        # Check if real query failed and pred returned empty or None
        real_failed_pred_empty = ('error_real' in ans_real and (ans_pred == '[]' or ans_pred == "['(None,)']"))
        
        if ans_pred!='null' and ans_pred is not None: # calculate the score over predicted answerable queries
            precision_ans_list.append(1 if ans_real not in ['null', None] else 0)
            precision_exec_list.append(1 if ans_real == ans_pred or both_failed or real_failed_pred_empty else 0)
        if ans_real!='null' and ans_real is not None: # calculate the score over GT answerable queries
            recall_ans_list.append(1 if ans_pred not in ['null', None] else 0)
            recall_exec_list.append(1 if ans_real == ans_pred or both_failed or real_failed_pred_empty else 0)
        acc_list.append(1 if ans_real == ans_pred or both_failed or real_failed_pred_empty else 0)
            
    precision_ans = sum(precision_ans_list) / (len(precision_ans_list)+1e-10)
    recall_ans = sum(recall_ans_list) / (len(recall_ans_list)+1e-10)
    f1_ans = 2*((precision_ans*recall_ans)/(precision_ans+recall_ans+1e-10))
    precision_exec = sum(precision_exec_list) / (len(precision_exec_list)+1e-10)
    recall_exec = sum(recall_exec_list) / (len(recall_exec_list)+1e-10)
    f1_exec = 2*((precision_exec*recall_exec)/(precision_exec+recall_exec+1e-10))
    acc = sum(acc_list) / (len(acc_list)+1e-10)

    out_eval = OrderedDict([
        ('precision_ans', round(100.0 * precision_ans, args.ndigits)),
        ('recall_ans', round(100.0 * recall_ans, args.ndigits)),
        ('f1_ans', round(100.0 * f1_ans, args.ndigits)),
        ('precision_exec', round(100.0 * precision_exec, args.ndigits)),
        ('recall_exec', round(100.0 * recall_exec, args.ndigits)),
        ('f1_exec', round(100.0 * f1_exec, args.ndigits)),
        ('acc', round(100.0 * acc, args.ndigits)),
    ])
    
    # Save overall metrics
    metrics_file = os.path.join(args.out_file, os.path.basename(args.pred_file).rsplit('.', 1)[0] + '_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(out_eval, f)

    # if args.out_file:
    #     if os.path.isfile(args.out_file):
    #         with open(args.out_file, 'a') as f:
    #             json.dump(out_eval, f)
    #             f.write('\n')
    #     else:
    #         with open(os.path.join(args.out_file, os.path.basename(args.pred_file).rsplit('.', 1)[0] + '.json'), 'w') as f:
    #             json.dump(out_eval, f)
    # else:
    #     print(json.dumps(out_eval, indent=2))

if __name__ == '__main__':
    args = parse_args()
    main(args)