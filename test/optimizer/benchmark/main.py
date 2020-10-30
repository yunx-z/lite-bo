import os
import sys
import time
import pickle
import argparse
from tabulate import tabulate
import numpy as np
from functools import partial

from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario


sys.path.append(os.getcwd())
from litebo.utils.load_data import load_data
from litebo.benchmark.search_space.build_cs import get_cs
from litebo.benchmark.objective_functions.eval_func import eval_func
from litebo.optimizer.smbo import SMBO
from litebo.optimizer.parallel_smbo import pSMBO
from litebo.optimizer.psmac import PSMAC


parser = argparse.ArgumentParser()
dataset_set = 'waveform-5000(1),waveform-5000(2),phoneme,page-blocks(1),page-blocks(2),pc2,optdigits,satimage,wind,' \
              'musk,delta_ailerons,mushroom,puma8NH,kin8nm,cpu_small,puma32H,cpu_act,bank32nh,visualizing_soil,mc1'

parser.add_argument('--datasets', type=str, default=dataset_set)
parser.add_argument('--algo_id', type=str, default='liblinear_svc,lightgbm,random_forest,adaboost')
parser.add_argument('--methods', type=str, default='litebo,smac,plitebo,psmac') # TODO: hyperopt
parser.add_argument('--rep_num', type=int, default=10)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--max_runs', type=int, default=50)

# save_dir = './data/benchmark_results/exp1/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

args = parser.parse_args()
dataset_list = args.datasets.split(',')
algo_ids = args.algo_id.split(',')
methods = args.methods.split(',')
rep_num = args.rep_num
start_id = args.start_id
seed = args.seed
max_runs = args.max_runs

np.random.seed(args.seed)

data_dir = './test/optimizer/data/cls_datasets'

def evaluate_smac(model,x,y,cs):
    scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                         "runcount-limit": max_runs,
                         "cs": cs,
                         "deterministic": "true",
                         "abort_on_first_run_crash": "false",
                         })
    eval = partial(eval_func, x=x, y=y, model=model)
    smac = SMAC(scenario=scenario, rng=np.random.RandomState(None), tae_runner=eval)
    incumbent = smac.optimize()
    inc_value = eval(incumbent)
    return inc_value

def evaluate(model,mth,x,y,dataset):
    cs = get_cs(algo_id)
    eval = partial(eval_func, x=x, y=y, model=model)
    inc_value=0
    start_time=time.time()
    for i in range(rep_num):
        print('=' * 10, "algo:", model, "| method:", mth, "| dataset:", dataset,"| rep_num",i,"="*10)
        if mth == 'litebo':
            bo = SMBO(eval, cs, max_runs=max_runs, time_limit_per_trial=60, logging_dir='logs')
            bo.run()
            inc_value += bo.get_incumbent()[0][1]
        elif mth == 'smac':
            inc_value += evaluate_smac(model,x,y,cs)
        elif mth == 'plitebo':
            bo = pSMBO(eval, cs, max_runs=max_runs, time_limit_per_trial=60, logging_dir='logs',
                       parallel_strategy='async', batch_size=4)
            bo.run()
            inc_value += bo.get_incumbent()[0][1]
        elif mth == 'sync_plitebo':
            bo = pSMBO(eval, cs, max_runs=max_runs, time_limit_per_trial=60, logging_dir='logs',
                       parallel_strategy='sync', batch_size=4)
            bo.run()
            inc_value += bo.get_incumbent()[0][1]
        elif mth == 'psmac':
            bo = PSMAC(eval, cs, n_jobs=4, evaluation_limit=max_runs, output_dir='logs')
            inc_perf, _, _ = bo.iterate()
            inc_value += inc_perf
        elif mth == 'hyperopt':
            pass
        else:
            raise ValueError('Invalid BO method - %s.' % mth)
    end_time=time.time()
    avg_time=(end_time-start_time)/rep_num
    inc_value/=rep_num
    return inc_value, avg_time

def check_datasets(datasets, data_dir):
    for _dataset in datasets:
        try:
            _ = load_data(_dataset, data_dir)
        except Exception as e:
            raise ValueError('Dataset - %s does not exist!' % _dataset)


if __name__ == "__main__":
    check_datasets(dataset_list,data_dir)
    table_data = list()
    table_header = ['Dataset', 'Algorithm']
    for mth in methods:
        table_header.append(mth + "_inc_value")
        table_header.append(mth + "_time")

    for dataset in dataset_list:
        _x, _y = load_data(dataset, data_dir)

        for algo_id in algo_ids:
            row=list()
            row.append(dataset)
            row.append(algo_id)
            for mth in methods:
                try:
                    inc_value,avg_time=evaluate(algo_id,mth,_x,_y,dataset)
                except:
                    row.append(-1)
                    row.append(-1)
                else:
                    row.append(inc_value)
                    row.append(avg_time)
            table_data.append(row)

            print(tabulate(table_data, headers=table_header, tablefmt='grid'))

