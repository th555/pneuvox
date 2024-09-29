""" adapted from https://github.com/EvolutionGym/evogym/tree/main/examples/cppn_neat """

import os
import shutil
import random
import numpy as np
import neat
from functools import partial
import pickle
import sys
import time
import random
import gzip
import pickle
from configparser import ConfigParser

save_dir = '/var/scratch/tmk390/saved_data'

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '..')
sys.path.insert(0, root_dir)

from parallel import ParallelEvaluator
from population import Population

from voxelbot import Voxelbot, VoxelbotGenome
from reproduction import MIPReproduction

from scipy.ndimage import zoom


def eval_genome_fitness(save_path, genome, config, genome_id, generation):
    robot = Voxelbot(genome, config)
    if save_path:
        save_path_generation = os.path.join(save_path, f'generation_{generation}')
        save_path_structure = os.path.join(save_path_generation, 'structure', f'{genome_id}')
        # save_path_controller = os.path.join(save_path_generation, 'controller')
        """ Don't save to save space on the DAS """
        # if not (generation % 10):
        #     # Only save all bots of every 10th generation
        #     with open(save_path_structure, 'wb') as f:
        #         pickle.dump(robot, f)

    return robot.fitness()

def eval_genome_constraint(genome, config, genome_id, generation):
    robot = Voxelbot(genome, config)
    validity = robot.valid and robot.actuatable
    return validity


# Copied from the Checkpointer class so it can use our own Population class
def restore_checkpoint(filename):
        """Resumes the simulation from a previous saved point."""
        with gzip.open(filename) as f:
            generation, config, population, species_set, rndstate = pickle.load(f)
            random.setstate(rndstate)
            return Population(config, (population, species_set, generation))


class SaveResultReporter(neat.reporting.BaseReporter):

    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self.generation = None

    def start_generation(self, generation):
        self.generation = generation
        save_path_structure = os.path.join(self.save_path, f'generation_{generation}', 'structure')
        # save_path_controller = os.path.join(self.save_path, f'generation_{generation}', 'controller')
        os.makedirs(save_path_structure, exist_ok=True)
        # os.makedirs(save_path_controller, exist_ok=True)

    def post_evaluate(self, config, population, species, best_genome):
        save_path_ranking = os.path.join(self.save_path, f'generation_{self.generation}', 'output.txt')
        genome_id_list, genome_list = np.arange(len(population)), np.array(list(population.values()))
        sorted_idx = sorted(genome_id_list, key=lambda i: genome_list[i].fitness, reverse=True)
        genome_id_list, genome_list = list(genome_id_list[sorted_idx]), list(genome_list[sorted_idx])
        with open(save_path_ranking, 'w') as f:
            out = ''
            for genome_id, genome in zip(genome_id_list, genome_list):
                if hasattr(genome, 'directed_locomotion') and genome.directed_locomotion:
                    out += f'{genome.key}\t\t{genome.fitness}\t\t{genome.fitness_distance}\n'
                else:
                    out += f'{genome.key}\t\t{genome.fitness}\n'
            f.write(out)


def run_cppn_neat(
        experiment_name,
        max_generations,
        num_cores,
        resume=None,
        mpi=False
    ):

    save_path = os.path.join(save_dir, experiment_name)

    if not resume:
        try:
            os.makedirs(save_path)
        except:
            print(f'THIS EXPERIMENT ({experiment_name}) ALREADY EXISTS')
            print('Override? (y/n): ', end='')
            ans = input()
            if ans.lower() == 'y':
                shutil.rmtree(save_path)
                os.makedirs(save_path)
            else:
                return None, None
            print()

        save_path_metadata = os.path.join(save_path, 'metadata.txt')
        with open(save_path_metadata, 'w') as f:
            f.write(f'MAX_GENERATIONS: {max_generations}\n')

        config_path = os.path.join(curr_dir, 'config_body.cfg')
        config = neat.Config(
            VoxelbotGenome,
            MIPReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
        pop = Population(config)


        save_path_config = os.path.join(save_path, 'config_body.cfg')
        shutil.copyfile(config_path, save_path_config)


    else:
        pop = resume

    reporters = [
        neat.StatisticsReporter(),
        neat.StdOutReporter(True),
        SaveResultReporter(save_path),
        neat.checkpoint.Checkpointer(1, None, os.path.join(save_path, 'checkpoint_'))
    ]
    for reporter in reporters:
        pop.add_reporter(reporter)

    evaluator = ParallelEvaluator(num_cores, partial(eval_genome_fitness, save_path), eval_genome_constraint)

    if mpi:
        fitness_function = evaluator.evaluate_fitness_mpi_master
    else:
        fitness_function = evaluator.evaluate_fitness

    pop.run(
        fitness_function=fitness_function,
        constraint_function=evaluator.evaluate_constraint,
        n=max_generations)

    best_robot = Voxelbot(pop.best_genome, config.genome_config)
    best_fitness = pop.best_genome.fitness
    return best_robot, best_fitness


def get_checkpoints_in_order(folder):
    generations = []
    for file in os.listdir(folder):
        if file.startswith('checkpoint_'):
            gen = {}
            n = int(file.split('_')[-1])
            
            gen['n'] = n
            checkpoint_path = os.path.join(folder, file)
            gen['checkpoint_path'] = checkpoint_path

            generations.append(gen)
    generations.sort(key=lambda d: d['n'])
    return generations


def wait_for_remote_jobs(evaluator):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    processor = MPI.Get_processor_name()
    print(f'This is worker process {rank} out of {size} on processor {processor}')
    while True:
        # receive
        angle, genomes, config, generation = comm.scatter(None, root=0)
        result_dict = evaluator.evaluate_fitness_mpi_worker(angle, genomes, config, generation)
        comm.gather((angle, result_dict), root=0)


if __name__ == '__main__':
    start = time.time()
    pop = None
    if len(sys.argv) > 2 and sys.argv[1] == "restore":
        """ e.g.
        python evo/run.py restore saved_data/ctrl_mutate_test/checkpoint_100
        """
        save_path = f'{save_dir}/{sys.argv[2]}'
        experiment_name = os.path.split(save_path)[1]
        gens = get_checkpoints_in_order(save_path)
        filename = gens[-1]['checkpoint_path']

        pop = restore_checkpoint(filename)
    elif len(sys.argv) > 1:
        experiment_name = sys.argv[1]
    else:
        print('Experiment name not given')
    num_cores = 50
    max_generations = 1000


    # Check if we're going to use MPI
    config = ConfigParser()
    config.read(os.path.join(curr_dir, 'config_body.cfg'))
    if config['VoxelbotGenome']['multi_direction_mpi'] == 'True':
        """ We are going to use MPI to evaluate multiple robots on different servers in parallel,
        this requires splitting up execution based on which server we find ourselves on. Only the root
        server will run the normal evolution code, the other servers will enter an alternate code
        path where they receive subpopulations, run simulations, and send the results back. """
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        processor = MPI.Get_processor_name()
        print(f'This is process {rank} out of {size} on processor {processor}')
        assert size == 5, 'Should be run with 5 mpi processes'
        if rank == 0:
            run_cppn_neat(
                experiment_name = experiment_name,
                max_generations = max_generations,
                num_cores = num_cores,
                resume = pop,
                mpi=True
            )
        else:
            # No save path on the workers, 1 is enough
            evaluator = ParallelEvaluator(num_cores, partial(eval_genome_fitness, ''), eval_genome_constraint)
            wait_for_remote_jobs(evaluator)
    else:
        run_cppn_neat(
            experiment_name = experiment_name,
            max_generations = max_generations,
            num_cores = num_cores,
            resume = pop,
            mpi=False
        )


    end = time.time()
    print(f'Elapsed time: {end - start}s')

