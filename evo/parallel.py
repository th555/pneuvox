""" source: https://github.com/EvolutionGym/evogym/tree/main/examples/cppn_neat
Runs evaluation functions in parallel subprocesses
in order to evaluate multiple genomes at once.
"""
import multiprocessing as mp
# from mpi4py import MPI

"""Apparent source: https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic"""
class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.

# class Pool(mp.pool.Pool):
#     Process = NoDaemonProcess

class NonDaemonPool(mp.pool.Pool):
    def Process(self, *args, **kwds):
        proc = super(NonDaemonPool, self).Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess
        return proc


class ParallelEvaluator(object):
    def __init__(self, num_workers, fitness_function, constraint_function=None, timeout=None):
        """
        fitness_function should take one argument, a tuple of
        (genome object, config object), and return
        a single float (the genome's fitness).
        constraint_function should take one argument, a tuple of
        (genome object, config object), and return
        a single bool (the genome's validity).
        """
        self.num_workers = num_workers
        self.fitness_function = fitness_function
        self.constraint_function = constraint_function
        self.timeout = timeout
        self.pool = NonDaemonPool(num_workers)

    def __del__(self):
        self.pool.close() # should this be terminate?
        self.pool.join()

    def evaluate_fitness_mpi_master(self, genomes, config, generation):
        for _, genome in genomes:
            assert hasattr(genome, 'directed_locomotion') and genome.directed_locomotion, 'Only in directed locomotion experiments'
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        processor = MPI.Get_processor_name()

        toscatter = [None]
        for angle in [-40, -20, 20, 40]:
            toscatter.append((angle, genomes, config, generation))
        scattered = comm.scatter(toscatter, root=0)

        jobs = []
        for i, (_, genome) in enumerate(genomes):
            jobs.append(self.pool.apply_async(self.fitness_function, (genome, config, i, generation)))

        # First start own jobs, then gather other jobs!

        gathered = comm.gather(None, root=0)

        result_dicts = []
        for result in gathered:
            if result is None:
                continue
            angle, result_dict = result
            result_dicts.append(result_dict)


        # assign the fitness back to each genome
        for i, (job, (genome_id, genome)) in enumerate(zip(jobs, genomes)):
            result_fitness, result_distance, result_killed, the_hash = job.get(timeout=self.timeout)
            # Get the results for different angles from the other servers
            for result_dict in result_dicts:
                angle_fitness, angle_distance, angle_killed = result_dict[genome_id]
                if not genome.fused_fitness:
                    result_fitness += angle_fitness
                result_distance += angle_distance
                result_killed = result_killed or angle_killed
            if result_killed:
                # Killed in one branch, total fitness zeroed too
                result_fitness = 0
                result_distance = 0
            else:
                if genome.fused_fitness:
                    Dpp0 = L = dB0B1 = Ppp1 = 0
                    for fit in [result_fitness] + [result_dict[genome_id][0]]:
                        Dpp0 += fit['Dpp0']
                        L += fit['L']
                        dB0B1 += fit['dB0B1']
                        Ppp1 += fit['Ppp1']
                    e = 0.00001
                    result_fitness = abs(Dpp0) / (L + e) * (Dpp0 / (dB0B1 + 1) - Ppp1)


            if the_hash in genome.fitness_cache:
                # TODO can be restructured so we don't run an entire simulation just to get the json hash
                cached_fitness, cached_distance, cached_killed = genome.fitness_cache[the_hash]
                genome.fitness = cached_fitness
                genome.fitness_distance = cached_distance
                genome.killed = cached_killed
            else:
                genome.fitness = result_fitness
                genome.fitness_distance = result_distance
                genome.killed = result_killed
                genome.fitness_cache[the_hash] = (result_fitness, result_distance, result_killed)
        

    def evaluate_fitness_mpi_worker(self, angle, genomes, config, generation):
        result_dict = {}
        jobs = []
        for i, (_, genome) in enumerate(genomes):
            genome.evaluate_angle = angle
            jobs.append(self.pool.apply_async(self.fitness_function, (genome, config, i, generation)))
        for i, (job, (genome_id, genome)) in enumerate(zip(jobs, genomes)):
            result_fitness, result_distance, result_killed, the_hash = job.get(timeout=self.timeout)
            result_dict[genome_id] = (result_fitness, result_distance, result_killed)
        return result_dict

    def evaluate_fitness(self, genomes, config, generation):
        jobs = []
        for i, (_, genome) in enumerate(genomes):
            jobs.append(self.pool.apply_async(self.fitness_function, (genome, config, i, generation)))

        # assign the fitness back to each genome
        for i, (job, (_, genome)) in enumerate(zip(jobs, genomes)):
            if hasattr(genome, 'directed_locomotion') and genome.directed_locomotion:
                result_fitness, result_distance, result_killed, the_hash = job.get(timeout=self.timeout)
                if the_hash in genome.fitness_cache:
                    # TODO can be restructured so we don't run an entire simulation just to get the json hash
                    cached_fitness, cached_distance, cached_killed = genome.fitness_cache[the_hash]
                    genome.fitness = cached_fitness
                    genome.fitness_distance = cached_distance
                    genome.killed = cached_killed
                else:
                    genome.fitness = result_fitness
                    genome.fitness_distance = result_distance
                    genome.killed = result_killed
                    genome.fitness_cache[the_hash] = (result_fitness, result_distance, result_killed)
            else:
                result_fitness, result_killed, the_hash = job.get(timeout=self.timeout)
                if the_hash in genome.fitness_cache:
                    # TODO can be restructured so we don't run an entire simulation just to get the json hash
                    cached_fitness, cached_killed = genome.fitness_cache[the_hash]
                    genome.fitness = cached_fitness
                    genome.killed = cached_killed
                else:
                    genome.fitness = result_fitness
                    genome.killed = result_killed
                    genome.fitness_cache[the_hash] = (result_fitness, result_killed)

    def evaluate_constraint(self, genomes, config, generation):
        validity_all = []
        for i, (_, genome) in enumerate(genomes):
            validity = self.constraint_function(genome, config, i, generation)
            validity_all.append(validity)
        return validity_all
