import numpy as np
import shutil
import sys
import csv
import os
from pprint import pprint
import argparse
import json
import pickle

import neat

from parallel import ParallelEvaluator
from population import Population
from voxelbot import Voxelbot, VoxelbotGenome

from run import restore_checkpoint, SaveResultReporter

from collections import defaultdict
from glob import glob
import itertools as it



def load_population(gen):
    """ in order of fitness (high to low)"""
    individuals = []
    pop = restore_checkpoint(gen['checkpoint_path']).population
    fitdict = {}
    with open(gen['results_path'], 'r') as f:
        for row in f:
            key, fitness = row.split()
            fitdict[int(key)] = float(fitness)
    for bot in pop.values():
        botdict = {}
        botdict['genome'] = bot
        import pdb; pdb.set_trace()
        botdict['fitness'] = fitdict[bot.key]
        individuals.append(botdict)
    individuals.sort(key=lambda d: -d['fitness'])
    pprint(individuals)
    return individuals


def load_fitnesses(gen):
    fitdict = {}
    distancedict = {}
    with open(gen['results_path'], 'r') as f:
        for row in f:
            splitrow = row.split()
            if len(splitrow) == 2:
                key, fitness = row.split()
                fitdict[int(key)] = float(fitness)
            elif len(splitrow) == 3:
                key, fitness, distance = row.split()
                fitdict[int(key)] = float(fitness)
                distancedict[int(key)] = float(distance)
    return fitdict, distancedict

def load_fitnesses2(results_path):
    fitdict = {}
    distancedict = {}
    with open(results_path, 'r') as f:
        for row in f:
            splitrow = row.split()
            if len(splitrow) == 2:
                key, fitness = row.split()
                fitdict[int(key)] = float(fitness)
            elif len(splitrow) == 3:
                key, fitness, distance = row.split()
                if distance == 'None':
                    distance = 0
                fitdict[int(key)] = float(fitness)
                distancedict[int(key)] = float(distance)
    return fitdict, distancedict


def get_generations_in_order(folder):
    generations = []
    for file in os.listdir(folder):
        if file.startswith('generation_'):
            gen = {}
            n = int(file.split('_')[-1])
            
            gen['n'] = n

            results_path = os.path.join(folder, f'generation_{n}', 'output.txt')
            gen['results_path'] = results_path
            if not os.path.exists(results_path):
                continue

            checkpoint_path = os.path.join(folder, file)
            gen['checkpoint_path'] = checkpoint_path


            structure_path = os.path.join(folder, f'generation_{n}', 'structure')
            gen['structure_path'] = structure_path

            generations.append(gen)
    generations.sort(key=lambda d: d['n'])
    return generations


def generate_plot_data(filename):
    os.makedirs('plotdata', exist_ok=True)

    paths = glob(f'/var/scratch/tmk390/saved_data/{filename}*')

    runs = {}
    for path in paths:
        runs[os.path.split(path)[1]] = get_generations_in_order(path)

    toplot = {}

    for name, run in runs.items():
        fitmeans = []
        fitbests = []
        dismeans = []
        disbests = []
        for gen in run:
            fitd, disd = load_fitnesses(gen)
            fit = list(fitd.values())
            fitbests.append(max(fit))
            fitmeans.append(np.mean(fit))
            if disd:
                dis = list(disd.values())
                disbests.append(max(dis))
                dismeans.append(np.mean(dis))
        if disbests:
            toplot[name] = {'mean fitness': fitmeans, 'best fitness': fitbests,
                            'mean distance': dismeans, 'best distance': disbests}
        else:
            toplot[name] = {'mean fitness': fitmeans, 'best fitness': fitbests}

    with open(f'plotdata/{filename}.json', 'w') as f:
        json.dump(toplot, f)

def stats(filenames):
    """ Is the difference between means statistically significant? """
    from scipy.stats import ttest_ind, mannwhitneyu
    for bestmean in ['mean fitness', 'best fitness']:
        print(bestmean)
        means = {}
        for filename in filenames:
            with open(f'plotdata/{filename}.json', 'r') as f:
                data = json.load(f)
            means[filename] = [row[bestmean][500] for row in data.values()]
        vals = list(means.values())
        print(np.mean(vals[0]), np.mean(vals[1]))
        print(ttest_ind(vals[0], vals[1]))
        print(mannwhitneyu(vals[0], vals[1]))

def plot_from_data(filenames, avg=False, legend=True):
    from matplotlib import pyplot as plt
    os.makedirs('plots', exist_ok=True)

    colours = iter(['slateblue', 'darkslateblue', 'red', 'maroon'])
    plt.rcParams["figure.figsize"] = (12, 9)

    for filename in filenames:
        with open(f'plotdata/{filename}.json', 'r') as f:
            toplot = json.load(f)
        if avg:
            means = defaultdict(list)

            cpfx = os.path.commonprefix(list(toplot.keys()))
            cpfx = '_'.join(cpfx.split('_')[:-1])
            num = len(toplot)
            # seqnames = list(list(toplot.values())[0].keys())
            seqnames = ["mean fitness", "best fitness"]
            for seqname in seqnames:
                for seqs in toplot.values():
                    means[seqname].append(seqs[seqname])
            for seqs in means.values():
                # shortest = min(len(x) for x in seqs)
                shortest = 500
                for i in range(len(seqs)):
                    seqs[i] = seqs[i][:shortest]
            for seqname in seqnames:
                data = np.mean(means[seqname], axis=0)
                data_std = np.std(means[seqname], axis=0)
                cpfx = cpfx.replace('div20_1ppl_open', 'open-loop'
                    ).replace('div20_1ppl_closed', 'closed-loop'
                    ).replace('mdv2_open', 'open-loop'
                    ).replace('mdv2_closed', 'closed-loop')
                lbl = f'{cpfx}, {seqname} (avg of {num} runs)'
                col = next(colours)
                plt.plot(range(len(data)), data, label=lbl, color=col)
                plt.fill_between(range(len(data)), data - data_std, data + data_std, color=col, alpha=0.2)
                if not legend:
                    plt.annotate(lbl, xy=(len(data), data[-1]), va="center")
        else:
            for runname, seqs in toplot.items():
                for seqname, data in seqs.items():
                    lbl = f'{runname} {seqname}'
                    plt.plot(range(len(data)), data, label=lbl)
                    if not legend:
                        plt.annotate(lbl, xy=(len(data), data[-1]), va="center")
    plt.xlabel('Generation')
    plt.ylabel('Fitness (straight-line distance in cm, 10s)')
    # plt.ylabel('Fitness (directed locomotion composite)')
    plt.title('Gait learning pneumatic soft robots, 10x10x10 voxels, population size 50')
    # plt.title('Directed locomotion-learning pneumatic soft robots, 10x10x10 voxels, population size 50')
    if legend:
        plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f'plots/{filename}.png', dpi=300)
    plt.show()

def get_best_robots(destsubdir, paths, generations):
    pre = '/var/scratch/tmk390/saved_data/'
    os.makedirs(f'bestbots/{destsubdir}', exist_ok=True)

    for fullpath in it.chain(*[glob(f'{os.path.join(pre, p)}*') for p in paths]):
        run = get_generations_in_order(fullpath)
        path = os.path.split(fullpath)[1]
        for gen_i in generations:
            gen = run[gen_i]
            fit, dis = load_fitnesses(gen)
            best_key = max(fit.items(), key=lambda x: x[1])[0]
            strucpath = os.path.join(gen['structure_path'], str(best_key))
            dst = f'bestbots/{destsubdir}/{path}_gen_{gen["n"]:04d}_struc_{best_key}'
            shutil.copyfile(strucpath, dst)
            print(dst)

def get_best_robots_from_checkpoint(destsubdir, paths, generations):
    pre = '/var/scratch/tmk390/saved_data/'
    os.makedirs(f'bestbots/{destsubdir}', exist_ok=True)
    for experiment_path in it.chain(*[glob(f'{os.path.join(pre, p)}*') for p in paths]):
        for gen_i in generations:
            results_path = os.path.join(experiment_path, f'generation_{gen_i}', 'output.txt')
            fit, dis = load_fitnesses2(results_path)
            best_key = max(fit.items(), key=lambda x: x[1])[0]
            
            p = f'{experiment_path}/checkpoint_{gen_i}'
            pop = restore_checkpoint(p)
            population = pop.population
            best_genome = population[best_key]
            bot = Voxelbot(best_genome, pop.config)

            path = os.path.split(experiment_path)[1]
            dst = f'bestbots/{destsubdir}/{path}_gen_{gen_i:04d}_struc_{best_key}'
            with open(dst, 'wb') as f:
                pickle.dump(bot, f)
            print(dst)

def get_latest_best(destsubdir, paths):
    pre = '/var/scratch/tmk390/saved_data/'
    os.makedirs(f'bestbots/{destsubdir}', exist_ok=True)
    for fullpath in it.chain(*[glob(f'{os.path.join(pre, p)}*') for p in paths]):
        run = get_generations_in_order(fullpath)
        path = os.path.split(fullpath)[1]
        gen = run[-2] # -2 cuz -1 might not have results yet
        fit, dis = load_fitnesses(gen)
        best_key = max(fit.items(), key=lambda x: x[1])[0]
        strucpath = os.path.join(gen['structure_path'], str(best_key))
        dst = f'bestbots/{destsubdir}/{path}_gen_{gen["n"]:04d}_struc_{best_key}'
        shutil.copyfile(strucpath, dst)
        print(dst)

def get_latest_best_from_checkpoint(destsubdir, paths):
    pre = '/var/scratch/tmk390/saved_data/'
    os.makedirs(f'bestbots/{destsubdir}', exist_ok=True)
    for experiment_path in it.chain(*[glob(f'{os.path.join(pre, p)}*') for p in paths]):
        nums = [int(p.split('_')[-1]) for p in glob(f'{experiment_path}/checkpoint_*')]
        if not nums:
            continue
        mx = max(nums) - 1

        results_path = os.path.join(experiment_path, f'generation_{mx}', 'output.txt')
        fit, dis = load_fitnesses2(results_path)
        best_key = max(fit.items(), key=lambda x: x[1])[0]
        
        p = f'{experiment_path}/checkpoint_{mx}'
        pop = restore_checkpoint(p)
        population = pop.population
        best_genome = population[best_key]
        bot = Voxelbot(best_genome, pop.config)

        path = os.path.split(experiment_path)[1]
        dst = f'bestbots/{destsubdir}/{path}_gen_{mx:04d}_struc_{best_key}'
        with open(dst, 'wb') as f:
            pickle.dump(bot, f)
        print(dst)



def get_latest_checkpoints(destsubdir, paths):
    pre = '/var/scratch/tmk390/saved_data/'
    os.makedirs(f'checkpoints/{destsubdir}', exist_ok=True)
    for experiment_path in it.chain(*[glob(f'{os.path.join(pre, p)}*') for p in paths]):
        nums = [int(p.split('_')[-1]) for p in glob(f'{experiment_path}/checkpoint_*')]
        if not nums:
            continue
        mx = max(nums) - 1
        p = f'{experiment_path}/checkpoint_{mx}'
        path = os.path.split(experiment_path)[1]
        dst = f'checkpoints/{destsubdir}/{path}_checkpoint_{mx:04d}'
        shutil.copyfile(p, dst)


def remove_pickles():
    print('removing pickles...')
    pre = '/var/scratch/tmk390/saved_data/'
    for experiment_path in glob(f'{pre}/*'):
        print(experiment_path)
        for generation_path in glob(f'{experiment_path}/generation_*/structure'):
            print(generation_path)
            shutil.rmtree(generation_path)

def clean_checkpoints():
    print('cleaning checkpoints...')
    pre = '/var/scratch/tmk390/saved_data/'
    for experiment_path in glob(f'{pre}/*'):
        print(experiment_path)
        nums = [int(p.split('_')[-1]) for p in glob(f'{experiment_path}/checkpoint_*')]
        if not nums:
            continue
        mx = max(nums)
        for num in nums:
            # also keep next-to-last one since that one has the results
            if (not (num%100)) or (num == mx) or (num == mx-1):
                print(f'not removing {num}')
                continue
            file = f'{experiment_path}/checkpoint_{num}'
            print(f'removing {file}')
            os.remove(file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--getbest', nargs='*', metavar=('folder', 'experiment'),
        help='get results from experiment(s) and store in folder')
    parser.add_argument('--gens', nargs='*', metavar=('generations'), type=int,
        help='use with --getbest to get the best individuals at specific generations')
    parser.add_argument('--genplotdata', metavar=('filename'),
        help='generate data to be plotted with --plotdata, filename is globbed to potentially get multiple experiment dirs (e.g. filename=run will get [run_01, run_02, run_03]')
    parser.add_argument('--plotdata', nargs='*', metavar='filename(s)',
        help='plot the data generated with --genplotdata')
    parser.add_argument('--stats', nargs='*', metavar='filename(s)',
        help='do some stats on data generated with --genplotdata')
    parser.add_argument('--avg', action='store_true', help='plot the average over all runs')
    parser.add_argument('--removepickles', action='store_true', help='remove all robot files (but keep checkpoints and fitness results)')
    parser.add_argument('--cleancheckpoints', action='store_true', help='keep 1 in every 20 checkpoints and the last one, remove the rest')
    parser.add_argument('--getlastcheckpoints', nargs='*', metavar=('folder', 'experiment'),
        help='get the latest checkpoint of an experiment and store in folder')

    args = parser.parse_args()

    if(args.getbest and len(args.getbest) >= 2):
        if(args.gens):
            # get_best_robots(args.getbest[0], args.getbest[1:], args.gens)
            get_best_robots_from_checkpoint(args.getbest[0], args.getbest[1:], args.gens)
        else:
            get_latest_best_from_checkpoint(args.getbest[0], args.getbest[1:])
    elif(args.getlastcheckpoints):
        get_latest_checkpoints(args.getlastcheckpoints[0], args.getlastcheckpoints[1:])
    elif(args.genplotdata):
        generate_plot_data(args.genplotdata)
    elif(args.plotdata):
        plot_from_data(args.plotdata, args.avg)
    elif(args.stats):
        stats(args.stats)
    elif(args.removepickles):
        remove_pickles()
    elif(args.cleancheckpoints):
        clean_checkpoints()
    else:
        parser.print_help()

