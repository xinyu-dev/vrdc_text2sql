import csv
import itertools
import json
import random
import re
import sqlite3
from typing import Union
import os, shutil
import pandas

def preprare_directory(temp, exist_ok=False):
    """
    Create a new directory and delete the old one if it exists
    :param temp: str: path to the directory
    """
    if os.path.exists(temp) and not exist_ok:
        # Remove the directory and all its contents
        shutil.rmtree(temp)
    # Recreate the directory
    os.makedirs(temp, exist_ok=exist_ok)


def postprocess_sql_query_from_markdown(response, current_time="2105-12-31 23:59:00") -> Union[str, None]:
    if response is None:
        return None
    if response.lower().strip() == "none":
        return None
    response = response.strip()
    
    if "```sql" not in response or "```" not in response:
        if response.endswith("<extra_id_1>"):
            response = response[:-len("<extra_id_1>")]
        if response.strip().lower() == "none":
            return None
        if not response.endswith(";"):
            response += ";" 
        response = response.replace("''", "'").replace('< =', '<=')
        response = response.replace("%y", "%Y").replace('%j', '%J')
        return response
    
    par = r"^.*```sql(.*)```.*$"
    match = re.search(par, response, re.DOTALL)
    if match is None:
        return None
    response = match.group(1).strip()
    if not response.endswith(";"):
        response += ";"
    response = response.replace("''", "'").replace('< =', '<=')
    response = response.replace("%y", "%Y").replace('%j', '%J')
    return response

def fix_parentheses_balance(sql: str) -> str:
    """
    Check the balance of "(" and ")" in SQL query.
    If unbalanced, detect "))" and keep just 1 ")".
    """
    if not sql:
        return sql
    
    # Count opening and closing parentheses
    open_count = sql.count("(")
    close_count = sql.count(")")
    
    # If already balanced, return as is
    if open_count == close_count:
        return sql
    
    # If more closing parentheses than opening, fix double closing parentheses
    if close_count > open_count:
        # Find all occurrences of "))" and replace with ")"
        sql = re.sub(r"\)\)", ")", sql)
        # Recalculate counts after fixing
        open_count = sql.count("(")
        close_count = sql.count(")")
        
        # If still unbalanced and we have more closing parentheses,
        # remove excess closing parentheses from the end
        while close_count > open_count:
            # Find the last closing parenthesis and remove it
            last_close_pos = sql.rfind(")")
            if last_close_pos != -1:
                sql = sql[:last_close_pos] + sql[last_close_pos + 1:]
                close_count -= 1
            else:
                break
    
    # If more opening parentheses than closing, add missing closing parentheses
    elif open_count > close_count:
        missing_closes = open_count - close_count
        sql += ")" * missing_closes
    
    return sql

class query(object):
    
    def __init__(self, db_file):
        
        self.db_meta, self.db_tabs, self.db_head = self._load_db(db_file)
        self.agg_op = ['', 'count', 'max', 'min', 'avg']
        self.cond_op = ['=', '>', '<', '>=', '<=']
    
    def __call__(self, sql_):
        '''
        select $$$ ### from *** where ===
        '''
        '''###'''
        mm_agg_col = []
        for itm in sql_['agg_col']:
            tt = self.db_tabs[itm[0]]
            hh = self.db_head[tt][itm[1]]
            mm_agg_col.append('.'.join([tt, hh]))
        mm_agg_col = ','.join(mm_agg_col)
        '''$$$'''
        if sql_['sel'] == 0:
            mm_agg = '{}'.format(mm_agg_col)
        elif sql_['sel'] == 1:
            mm_agg = 'COUNT ( DISTINCT {} )'.format(mm_agg_col)
        elif sql_['sel'] == 2:
            mm_agg = 'MAX ( {} )'.format(mm_agg_col)
        elif sql_['sel'] == 3:
            mm_agg = 'MIN ( {} )'.format(mm_agg_col)
        elif sql_['sel'] == 4:
            mm_agg = 'AVG ( {} )'.format(mm_agg_col)
        '''***'''
        tbtb = [self.db_tabs[k] for k in sql_['table']]
        mm_tab = [tbtb[0]]
        for k in range(1, len(tbtb)):
            mm_tab.append('INNER JOIN')
            mm_tab.append(tbtb[k])
            mm_tab.append('on')
            mm_tab.append('{}.{} = {}.{}'.format(tbtb[0], 'HADM_ID', tbtb[k], 'HADM_ID'))
        '''==='''
        mm_cond = []
        for itm in sql_['cond']:
            tt = self.db_tabs[itm[0]]
            cc = self.db_head[tt][itm[1]]
            oo = self.cond_op[itm[2]]
            ff = itm[3]
            cond1 = '{}.{} {} {}'.format(tt, cc, oo, '"'+str(ff)+'"')
            mm_cond.append(cond1)
        mm_cond = ' AND '.join(mm_cond)
        bb_query = 'SELECT {} FROM {} WHERE {}'.format(mm_agg, ' '.join(mm_tab), mm_cond)
                
        return bb_query
    
    def _load_db(self, db_file):
        
        self.conn = sqlite3.connect(db_file)
        self.cur = self.conn.cursor()
        self.cur.execute("select * from sqlite_master;")
        results = self.cur.fetchall()
        db_meta = {}
        db_tabs = []
        db_head = {}
        for tb in results:
            db_meta[tb[2]] = {}
            db_tabs.append(tb[2])
            db_head[tb[2]] = {}
            arr = re.split('\n', tb[-1])[1:-1]
            dbaa = []
            for itm in arr:
                ttl = re.split('\s', itm)
                ttl = list(filter(None, ttl))
                db_meta[tb[2]][ttl[0]] = ttl[1]
                dbaa.append(ttl[0])
            db_head[tb[2]] = dbaa

        return (db_meta, db_tabs, db_head)
    
    def execute_sql(self, sql_):
        return self.cur.execute(sql_)

def get_value_pool_(db_file, model, samp_cond):
    (db_meta, db_tabs, db_head) = model._load_db(db_file)
    pool_ = []
    for itm in samp_cond:
        mytb = db_tabs[itm[0]]
        myhd = db_head[mytb][itm[1]]
        mysql = 'select {} from {}'.format(myhd, mytb)
        myres = model.execute_sql(mysql).fetchall()
        myres = list({k[0]: {} for k in myres})
        pool_.append(myres)
        
    return pool_