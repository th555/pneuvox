import neat
import numpy as np
import json
import os
import subprocess
import uuid
import itertools as it
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes, measurements, sum as ndi_sum, find_objects
import random

from genes import PneuvoxConnectionGene, PneuvoxNodeGene
from neat.config import ConfigParameter

import functools as ft
import operator as op
from collections import namedtuple


def mutate_gauss(genes, sds, limits=(-5, 5)):
    mutated = []
    for g, sd in zip(genes, sds):
        newg = random.gauss(g, sd)
        newg = max(limits[0], newg)
        newg = min(limits[1], newg)
        mutated.append(newg)
    return mutated


class VoxelbotGenomeConfig(neat.genome.DefaultGenomeConfig):
    """ Adding some custom config parameters """
    def __init__(self, params):
        super().__init__(params)
        self._extra_params = [
            ConfigParameter('control_type', str, 'recurrent_open'),
            ConfigParameter('bbox_x', int, 10),
            ConfigParameter('bbox_y', int, 10),
            ConfigParameter('bbox_z', int, 10),
            ConfigParameter('disable_osc', bool, False),
            ConfigParameter('init_recurrent_weights', str, 'zeroes'),
            ConfigParameter('terrain_type', str, 'flat'),
            ConfigParameter('terrain_seed', int, -1),
            ConfigParameter('ctrl_hidden_nodes', int, 10),
            ConfigParameter('eval_seconds', int, 10),
            ConfigParameter('crossover_method', str, 'random'),
            ConfigParameter('weight_init_sd', float, 2),
            ConfigParameter('weight_mut_init', float, 1),
            ConfigParameter('directed_locomotion', bool, False),
            ConfigParameter('fused_fitness', bool, False),
        ]
        for p in self._extra_params:
            setattr(self, p.name, p.interpret(params))


class VoxelbotGenome(neat.DefaultGenome):
    """
    Wraps the NEAT defaultgenome and adds an additional sub-genome for the controller
    """

    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = PneuvoxNodeGene
        param_dict['connection_gene_type'] = PneuvoxConnectionGene
        return VoxelbotGenomeConfig(param_dict)

    def __init__(self, key):
        super().__init__(key)
        self.n_hidden = 10 # may be changed in configure_new

        self.fitness_distance = 0

        # TODO n_recurrent should be either 0 or equal to n_hidden, so it should be changed into a simple flag for clarity
        self.n_recurrent = 0 # may be changed in configure_new

        self.max_actuators = 10 # maximum number of actuators that the robot supports,
                                # used to determine n_out of the controller
        self.freq_min = 1
        self.freq_max = 5

        # Whether this bot was killed (due to tipping over or 'exploding') in the last fitness evaluation
        self.killed = None

        self.fitness_cache = {}
        self.associations = {} # For associative crossover

        self.evaluate_angle = None # to be assigned by the MPI evaluator on a throwaway basis

        self.weights_names = [
            ('weights_hidden', 'weights_hidden_mut'),
            ('bias_hidden', 'bias_hidden_mut'),
            ('weights_out', 'weights_out_mut'),
            ('bias_out', 'bias_out_mut'),
            ('weights_recurrent', 'weights_recurrent_mut'),
            ('weights_recurrent_self', 'weights_recurrent_self_mut'),
            ('weights_context', 'weights_context_mut'),
            ('bias_context', 'bias_context_mut'),
            ('freq', 'freq_mut'),
        ]

    def configure_new(self, config):
        super().configure_new(config)
        self.config = config

        self.terrain_type = config.terrain_type
        self.terrain_seed = config.terrain_seed

        self.directed_locomotion = config.directed_locomotion if hasattr(config, 'directed_locomotion') else False
        self.fused_fitness = config.fused_fitness if hasattr(config, 'fused_fitness') else False

        self.control_type = config.control_type
        self.n_hidden = config.ctrl_hidden_nodes
        self.disable_osc = config.disable_osc
        self.crossover_method = config.crossover_method

        if self.control_type == 'recurrent_closed':
            self.n_recurrent = self.n_hidden # separate recurrent inputs
            self.n_in = 2 * (not self.disable_osc) + self.max_actuators # excluding the recurrent inputs
            self.n_out = self.max_actuators * 2 # Same as closed_loop_b
        elif self.control_type == 'recurrent_open':
            self.n_recurrent = self.n_hidden
            self.n_in = 2 #disable_osc has no effect here
            self.n_out = self.max_actuators * 2
        if self.directed_locomotion:
            self.n_in += 1 # for the direction sensor input


        self.freq = [random.uniform(self.freq_min, self.freq_max)]
        self.freq_mut = [0.2]

        mut_sd = self.config.weight_init_sd # initial strength of weight mutations

        self.weights_hidden = self.rndweights(self.n_in*self.n_hidden)
        self.weights_hidden_mut = [mut_sd] * len(self.weights_hidden)
        if(config.init_recurrent_weights == 'zeroes'):
            # Recurrent weights are initialized to 0, otherwise the starting behaviour may be too chaotic/random
            self.weights_recurrent = [0] * (self.n_recurrent*self.n_hidden)
            self.weights_recurrent_mut = [mut_sd] * len(self.weights_recurrent)
            self.weights_recurrent_self = [0] * self.n_recurrent
            self.weights_recurrent_self_mut = [mut_sd] * len(self.weights_recurrent_self)
            self.bias_context = [0] * (self.n_hidden)
            self.bias_context_mut = [mut_sd] * len(self.bias_context)
        elif(config.init_recurrent_weights == 'random'):
            self.weights_recurrent = self.rndweights(self.n_recurrent*self.n_hidden)
            self.weights_recurrent_mut = [mut_sd] * len(self.weights_recurrent)
            self.weights_recurrent_self = self.rndweights(self.n_recurrent)
            self.weights_recurrent_self_mut = [mut_sd] * len(self.weights_recurrent_self)
            self.bias_context = self.rndweights(self.n_hidden)
            self.bias_context_mut = [mut_sd] * len(self.bias_context)
        self.weights_out = self.rndweights(self.n_hidden*self.n_out)
        self.weights_out_mut = [mut_sd] * len(self.weights_out)
        self.bias_hidden = self.rndweights(self.n_hidden)
        self.bias_hidden_mut = [mut_sd] * len(self.bias_hidden)
        self.bias_out = self.rndweights(self.n_out)
        self.bias_out_mut = [mut_sd] * len(self.bias_out)
        self.weights_context = self.rndweights(self.n_hidden)
        self.weights_context_mut = [mut_sd] * len(self.weights_context)

        """ Voxelbot body parameters """
        self.bbox_input = [config.bbox_x, config.bbox_y, config.bbox_z]

        self.eval_seconds = config.eval_seconds

    def pack_ctrl_genome(self):
        """ Collapse all controller weights and mutation strengths into a single list of 2-tuples """
        packed = []
        packlens = []
        for name, name_m in self.weights_names:
            packed.extend(list(zip(getattr(self, name), getattr(self, name_m))))
            packlens.append(len(getattr(self, name)))
        return packed, packlens

    def unpack_ctrl_genome(self, packed, packlens):
        """ Unpack a packed ctrl genome and assign to self """
        packed = packed.copy()
        for (name, name_m), packlen in zip(self.weights_names, packlens):
            setattr(self, name, [])
            setattr(self, name_m, [])
            for i in range(packlen):
                vw, vmut = packed.pop(0)
                getattr(self, name).append(vw)
                getattr(self, name_m).append(vmut)

    def crossover_controller(self, genome1, genome2):
        assert genome1.n_in == genome2.n_in
        assert genome1.n_out == genome2.n_out

        self.control_type = genome1.control_type
        self.disable_osc = genome1.disable_osc
        self.terrain_type = genome1.terrain_type
        self.terrain_seed = genome1.terrain_seed
        self.directed_locomotion = genome1.directed_locomotion if hasattr(genome1, 'directed_locomotion') else False
        self.fused_fitness = genome1.fused_fitness if hasattr(genome1, 'fused_fitness') else False
        self.n_in = genome1.n_in
        self.n_out = genome1.n_out
        self.n_recurrent = genome1.n_recurrent
        self.n_hidden = genome1.n_hidden
        self.crossover_method = genome1.crossover_method

        """ Inherit voxelbot body parameters """
        self.bbox_input = genome1.bbox_input

        self.eval_seconds = genome1.eval_seconds


        # inherit weights, biases etc..
        """ The weights are structured like [in1-out1, in1-out2, in1-out3, ...] """
        if self.crossover_method == 'random':
            packed1, packlens = genome1.pack_ctrl_genome()
            packed2, _ = genome2.pack_ctrl_genome()

            packednew = []
            for a, b in zip(packed1, packed2):
                if random.random() < 0.5:
                    packednew.append(a)
                else:
                    packednew.append(b)
            self.unpack_ctrl_genome(packednew, packlens)
        elif self.crossover_method == '1pointpl':
            # 1-point per layer
            for name, name_m in self.weights_names:
                wts1 = getattr(genome1, name)
                muts1 = getattr(genome1, name_m)
                wts2 = getattr(genome2, name)
                muts2 = getattr(genome2, name_m)
                if random.random() < 0.5:
                    wts1, wts2 = wts2, wts1
                    muts1, muts2 = muts2, muts1
                p = random.randint(0, len(wts1))
                setattr(self, name, wts1[:p] + wts2[p:])
                setattr(self, name_m, muts1[:p] + muts2[p:])
        elif self.crossover_method == '1point':
            packed1, packlens = genome1.pack_ctrl_genome()
            packed2, _ = genome2.pack_ctrl_genome()
            if random.random() < 0.5:
                packed1, packed2 = packed2, packed1
            p = random.randint(0, len(packed1))
            packednew = packed1[:p] + packed2[p:]
            self.unpack_ctrl_genome(packednew, packlens)
        elif self.crossover_method == 'associative':
            if genome1 == genome2:
                """ Simply copy the controller and don't increment association counters """
                self.unpack_ctrl_genome(*genome1.pack_ctrl_genome())
                return

            packed1, packlens = genome1.pack_ctrl_genome()
            packed2, _ = genome2.pack_ctrl_genome()
            size = len(packed1)

            mygenes = {**self.nodes, **self.connections}
            genes1 = {**genome1.nodes, **genome1.connections}
            genes2 = {**genome2.nodes, **genome2.connections}

            """ First pick the top 50% of genes based on the associations """
            # Cumulative association values for controller genes from respective genomes based on the association lists of the selected genes
            cum_assoc_1 = np.zeros(size, dtype=int)
            cum_assoc_2 = np.zeros(size, dtype=int)
            """ copy configure_crossover from genome and mark genes w.r.t. origin """
            for gene in mygenes.values():
                if gene.associations is not None:
                    if gene.parent == genome1:
                        cum_assoc_1 += gene.associations
                    elif gene.parent == genome2:
                        cum_assoc_2 += gene.associations
                    else:
                        raise Exception("Gene without parents..")
                else:
                    gene.associations = np.zeros(size, dtype=int)

            method = 'highest'

            if method == 'highest': # top 50 % associative
                # Find the locations (in ctrl genome) with highest associations
                max_assoc = np.maximum(cum_assoc_1, cum_assoc_2)
                # The locations that will be used for assoc crossover
                do_assoc = max_assoc >= np.median(max_assoc)
                from_parent_1 = np.zeros(size, dtype=bool)
                for i in range(size):
                    if do_assoc[i] and (cum_assoc_1[i] != cum_assoc_2[i]):
                        if cum_assoc_1[i] > cum_assoc_2[i]:
                            from_parent_1[i] = 1
                    else:
                        from_parent_1[i] = random.random() < 0.5
            elif method == 'random': # random 75% associative
                from_parent_1 = np.zeros(size, dtype=bool)
                for i in range(size):
                    if (random.random() < 0.75) and (cum_assoc_1[i] != cum_assoc_2[i]):
                        if cum_assoc_1[i] > cum_assoc_2[i]:
                            from_parent_1[i] = 1
                    else:
                        from_parent_1[i] = random.random() < 0.5

            """ Build the new control genome """
            packednew = []
            for i, p1 in enumerate(from_parent_1):
                if p1:
                    packednew.append(packed1[i])
                else:
                    packednew.append(packed2[i])

            """ Update gene associations: at each gene, the locations where control
            genes were inherited from the same parent as itself are incremented. The others
            are set to 0. """
            from_parent_2 = np.invert(from_parent_1)
            for gene in mygenes.values():
                if gene.parent == genome1:
                    gene.associations *= from_parent_1
                    gene.associations[from_parent_1] += 1
                elif gene.parent == genome1:
                    gene.associations *= from_parent_2
                    gene.associations[from_parent_2] += 1

            self.unpack_ctrl_genome(packednew, packlens)
            # array = np.stack(list([gene.associations for gene in mygenes.values()])); print(array)
        else:
            raise Exception("Invalid crossover_method")

        # Combine fitness caches of both parents
        self.fitness_cache = {**genome1.fitness_cache, **genome2.fitness_cache}
        # Cache only the top 10 highest fitnesses
        sorted_ = sorted(self.fitness_cache.items(), key=lambda x: x[1][0], reverse=True)
        self.fitness_cache = dict(sorted_[:10])


    def configure_crossover(self, genome1, genome2, config, cross_morphology=True, cross_controller=True):
        if config.crossover_method == 'associative':
            # Set (pre-crossover) parent of all genes
            genes1 = {**genome1.nodes, **genome1.connections}
            genes2 = {**genome2.nodes, **genome2.connections}
            for gene in genes1.values():
                gene.parent = genome1
            for gene in genes2.values():
                gene.parent = genome2

        if cross_morphology:
            super().configure_crossover(genome1, genome2, config)
        else:
            """ No morphology crossover, so inherit the morphology from one parent """
            """ Inherit morphology from fittest parent """
            if genome1.fitness > genome2.fitness:
                mparent = genome1
            else:
                mparent = genome2
            super().configure_crossover(mparent, mparent, config)

        if cross_controller:
            self.crossover_controller(genome1, genome2)
        else:
            """ Inherit controller from fittest parent """
            if genome1.fitness > genome2.fitness:
                cparent = genome1
            else:
                cparent = genome2
            self.crossover_controller(cparent, cparent)

    def mutate(self, config, mutate_morphology=True, mutate_controller=True):
        if mutate_morphology:
            super().mutate(config)
        if mutate_controller:
            self.weights_hidden_mut = mutate_gauss(self.weights_hidden_mut, self.weights_hidden_mut)
            self.weights_hidden = mutate_gauss(self.weights_hidden, self.weights_hidden_mut)
            self.weights_out_mut = mutate_gauss(self.weights_out_mut, self.weights_out_mut)
            self.weights_out = mutate_gauss(self.weights_out, self.weights_out_mut)
            self.weights_recurrent_mut = mutate_gauss(self.weights_recurrent_mut, self.weights_recurrent_mut)
            self.weights_recurrent = mutate_gauss(self.weights_recurrent, self.weights_recurrent_mut)
            self.weights_recurrent_self_mut = mutate_gauss(self.weights_recurrent_self_mut, self.weights_recurrent_self_mut)
            self.weights_recurrent_self = mutate_gauss(self.weights_recurrent_self, self.weights_recurrent_self_mut)
            self.bias_hidden_mut = mutate_gauss(self.bias_hidden_mut, self.bias_hidden_mut)
            self.bias_hidden = mutate_gauss(self.bias_hidden, self.bias_hidden_mut)
            self.bias_out_mut = mutate_gauss(self.bias_out_mut, self.bias_out_mut)
            self.bias_out = mutate_gauss(self.bias_out, self.bias_out_mut)
            self.bias_context_mut = mutate_gauss(self.bias_context_mut, self.bias_context_mut)
            self.bias_context = mutate_gauss(self.bias_context, self.bias_context_mut)
            self.weights_context_mut = mutate_gauss(self.weights_context_mut, self.weights_context_mut)
            self.weights_context = mutate_gauss(self.weights_context, self.weights_context_mut)

            self.freq_mut = mutate_gauss(self.freq_mut, self.freq_mut)
            self.freq = mutate_gauss(self.freq, self.freq_mut)
            self.freq[0] = max(self.freq_min, self.freq[0])
            self.freq[0] = min(self.freq_max, self.freq[0])

    def save(self, f):
        raise NotImplementedError

    def rndweights(self, n):
        return mutate_gauss([0]*n, [self.config.weight_init_sd]*n)

    def controller_config(self):
        return {
            'freq': self.freq[0],
            'n_in': self.n_in,
            'n_hidden': self.n_hidden,
            'n_out': self.n_out,
            'n_recurrent': self.n_recurrent,
            'weights_hidden': self.weights_hidden,
            'bias_hidden': self.bias_hidden,
            'weights_out': self.weights_out,
            'bias_out': self.bias_out,
            'weights_recurrent': self.weights_recurrent,
            'weights_recurrent_self': self.weights_recurrent_self,
            'weights_context': self.weights_context,
            'bias_context': self.bias_context,
            'disable_osc': int(self.disable_osc),
        }



class Voxelbot:
    """ Class that generates the phenotype from the genome using
    CPPN, passes the phenotype to the simulator, runs experiments
    to evaluate fitness, etc.. """
    def __init__(self, genome, config, debug_cppn=None, donttrim=False):
        self.genome = genome
        self.config = config
        self.bbox_input = self.genome.bbox_input # voxels per side
        self.voxsize = 0.01 #size of single voxel (in meters)
        self.n_materials = 10 # Number of discrete materials into which to divide the range of stiffness values
        self.pneu_indices = []

        if debug_cppn:
            self.cppn = debug_cppn
        else:
            self.cppn = neat.nn.FeedForwardNetwork.create(self.genome, self.config)
        self.valid = False
        self.actuatable = False
        self.bbox = None
        self.voxels = None

        self.eval_cppn(donttrim)

    def bin_materials(self, mat):
        """ Convert a continuous stiffness output of the CPPN, between -1 and 1, to one of
        n_materials discrete materials. """
        return np.digitize(np.clip(mat, -1, 1), np.linspace(-1, 1, 10)) # digitize goes from 1 to 10 inclusive

    def eval_cppn(self, donttrim=False):
        voxels = np.zeros(self.bbox_input, dtype=int)
        sorting = np.zeros(self.bbox_input, dtype=float)
        pneumatics = np.zeros_like(voxels)
        for i, x in enumerate(np.linspace(0, 1, self.bbox_input[0])):
            for j, y in enumerate(np.linspace(0, 1, self.bbox_input[1])):
                for k, z in enumerate(np.linspace(0, 1, self.bbox_input[2])):
                    # Use distance from center as an extra input
                    d = np.linalg.norm(np.array([x, y, z]) - np.array(self.bbox_input)/2)
                    presence, is_pneu, stiffness, sorting_i = self.cppn.activate((x, y, z, d))
                    sorting[i, j, k] = sorting_i

                    if presence > 0.1:
                        if is_pneu > 0.1:
                            pneumatics[i, j, k] = 1
                        else:
                            voxels[i, j, k] = self.bin_materials(stiffness)

        if not np.any(voxels):
            self.valid = False
            return
        self.valid = True


        # Detect holes in the structural voxels and add these to pneumatics
        struct_holes = binary_fill_holes(voxels).astype(int) - voxels.astype(bool)
        if np.any(struct_holes):
            # print("Struct holes!")
            pneumatics = pneumatics | struct_holes

        # New pneu strategy: binary_fill_holes in pneu, find pneu inner edges using erosion (and XOR or smth), add those to voxels using correct cppn structural output. After that take the largest component.

        # Ensure (actively) the pneumatic voxels are always surrounded by structural voxels
        # Iterations=2 ensures the walls are always at least 2 voxels thick, this is not done at the moment
        # Generate shell on the inside of the pneu areas
        pneumatics = binary_fill_holes(pneumatics)
        pneu_erosion = binary_erosion(pneumatics, iterations=1, structure=np.ones((3,3,3)))
        pneumatics = pneu_erosion # use the eroded version from now on
        pneu_shell = binary_dilation(pneumatics, iterations=1, structure=np.ones((3,3,3))).astype(int) - pneumatics.astype(bool) #dilation and erosion are not exact complements, so we generate the actual shell by dilation of the eroded voxels

        # replace the shell with structural voxels according to the CPPN
        for i, x in enumerate(np.linspace(0, 1, self.bbox_input[0])):
            for j, y in enumerate(np.linspace(0, 1, self.bbox_input[1])):
                for k, z in enumerate(np.linspace(0, 1, self.bbox_input[2])):
                    if pneu_shell[i, j, k]:
                        d = np.linalg.norm(np.array([x, y, z]) - np.array(self.bbox_input)/2)
                        presence, is_pneu, *stiffness, sorting_i = self.cppn.activate((x, y, z, d))
                        voxels[i, j, k] = self.bin_materials(stiffness) # see above

        # Find largest connected component and discard the rest
        blobs, nblobs = measurements.label(
            voxels.astype(bool), # bool is needed because otherwise it considers different materials as different components
        )

        areas = ndi_sum(voxels.astype(bool), blobs, range(1, np.max(blobs)+1))
        largest_label = np.argmax(areas) + 1
        mask = binary_fill_holes(blobs == largest_label)

        voxels = voxels * (mask)
        pneumatics = pneumatics * (mask)

        if donttrim:
            self.bbox = self.bbox_input.copy()
        else:
            # Trim to bounding box (mostly so it doesn't start in mid-air)
            xslice, yslice, zslice = find_objects(mask)[0]
            x1 = xslice.start
            x2 = xslice.stop
            y1 = yslice.start
            y2 = yslice.stop
            z1 = zslice.start
            z2 = zslice.stop

            voxels = voxels[x1:x2, y1:y2, z1:z2]
            sorting = sorting[x1:x2, y1:y2, z1:z2]
            pneumatics = pneumatics[x1:x2, y1:y2, z1:z2]
            self.bbox = [x2-x1, y2-y1, z2-z1]


        """post-process pneumatics:
            - Separate in blobs
            - Sort based on the 'sorting' output of the CPPN
            - Check if each is fully enclosed by structural voxels
            - Add valid pneumatics to the voxel array and make sure the exporter knows about it
        """

        pneu_chambers = []
        pneublobs, npneublobs = measurements.label(pneumatics.astype(bool), structure=np.ones((3,3,3)))

        order = []
        if npneublobs: # Sort according to 'sorting' output at centroids of pneu chambers
            labels = list(range(1, npneublobs+1))
            centers = measurements.center_of_mass(pneublobs.astype(bool), pneublobs, labels)
            order_ = sorted(zip(labels, centers), key=lambda l_s: sorting[(*map(round, l_s[1]),)])
            order = list([o[0] for o in order_])

        pneu_i = self.n_materials + 1
        for i in range(npneublobs)[:self.genome.max_actuators]:
            pneu_label = order[i]
            self.actuatable = True
            thispneu = (pneublobs == pneu_label)
            voxels[thispneu] = pneu_i
            self.pneu_indices.append(pneu_i)
            pneu_i += 1

        self.voxels = voxels
        return voxels

    def preprocess_voxels(self, voxels):
        for ind in self.pneu_indices:
            voxels[voxels==ind] = 0
        return voxels

    def voxel_difference(self, voxels1, voxels2):
        """ Return the fraction of voxels (between 0.0 and 1.0) that changed """
        assert voxels1.shape == voxels2.shape
        voxels1 = self.preprocess_voxels(voxels1)
        voxels2 = self.preprocess_voxels(voxels2)
        diff = np.sum(voxels1.astype(bool) != voxels2.astype(bool)) / ft.reduce(op.mul, voxels1.shape)
        # print(f'difference: {diff}')
        return diff

    def get_pneu_quads(self, wallvox, pneuvox, plane, normal):
        """ For each voxel:
         - check if it is the "bottom-left" corner of a quad
         - check if it has a pneu voxel in the given normal direction
         - if so, add it to the quads list

        TODO: refactor?
        """
        dx, dy, dz = plane
        quads = []
        for x in range(self.bbox[0]):
            for y in range(self.bbox[1]):
                for z in range(self.bbox[2]):
                    if wallvox[x, y, z]:
                        isquad = True
                        if dx:
                            try:
                                isquad &= wallvox[x+dx, y, z]
                            except IndexError:
                                pass
                        else:
                            try:
                                isquad &= wallvox[x, y+dy, z+dz]
                            except IndexError:
                                pass
                        if dy:
                            try:
                                isquad &= wallvox[x, y+dy, z]
                            except IndexError:
                                pass
                        else:
                            try:
                                isquad &= wallvox[x+dx, y, z+dz]
                            except IndexError:
                                pass
                        if dz:
                            try:
                                isquad &= wallvox[x, y, z+dz]
                            except IndexError:
                                pass
                        else:
                            try:
                                isquad &= wallvox[x+dx, y+dy, z]
                            except IndexError:
                                pass

                        if isquad:
                            try:
                                if not dx:
                                    ys = [y, y+dy]
                                    zs = [z, z+dz]
                                    indices = [(x+normal, y, z) for y, z in it.product(ys, zs)]
                                    if any(pneuvox[ind] for ind in indices):
                                        quads.append([(x, y, z), (x, y+dy, z), (x, y+dy, z+dz), (x, y, z+dz)])
                                elif not dy:
                                    xs = [x, x+dx]
                                    zs = [z, z+dz]
                                    indices = [(x, y+normal, z) for x, z in it.product(xs, zs)]
                                    if any(pneuvox[ind] for ind in indices):
                                        quads.append([(x, y, z), (x+dx, y, z), (x+dx, y, z+dz), (x, y, z+dz)])
                                elif not dz:
                                    xs = [x, x+dx]
                                    ys = [y, y+dy]
                                    indices = [(x, y, z+normal) for x, y in it.product(xs, ys)]
                                    if any(pneuvox[ind] for ind in indices):
                                        quads.append([(x, y, z), (x+dx, y, z), (x+dx, y+dy, z), (x, y+dy, z)])
                            except IndexError:
                                pass
        return quads

    def make_pneu_quads(self, i):
        pneu_voxels = self.voxels == i
        wall_voxels = binary_dilation(pneu_voxels, structure=np.ones((3,3,3))).astype(int) - pneu_voxels.astype(bool)

        # print(pneu_voxels.astype(int))

        # DEBUG check wall voxels for overlap with pneumatic voxels
        for pneu_i in self.pneu_indices:
            pneu_stuff = self.voxels == pneu_i
            overlap = pneu_stuff & wall_voxels.astype(bool)
            if(np.any(overlap)):
                print("ERROR ER IS OVERLAP")

        quads = []
        # Treat separately each plane (XY, XZ etc..) and both normal directions per plane
        for plane, direction in [
            [(1, 1, 0), 1], # XY, positive Z
            [(1, -1, 0), -1], # XY, negative Z
            [(1, 0, 1), -1], # XZ, negative Y
            [(1, 0, -1), 1], # XZ, positive Y
            [(0, 1, 1), 1], # YZ, positive X
            [(0, 1, -1), -1] # YZ, negative X
        ]:
            quads.extend(self.get_pneu_quads(wall_voxels, pneu_voxels, plane, direction))

        return quads

    def export(self, filename, eval_seconds, video_filename, pictures_filename, extra_info):
        """ File includes:
         - bbox (size of the cube in voxels per size)
         - voxel size in mm
         - material specifications and their indices:
            - 0: empty
            - 1, 2, etc..: stiffness, mass
            - 3: pneu
         - the voxels (one number per line)

        Returns a hash of the exported json for caching purposes
        """
        data = dict()
        data['eval_seconds'] = eval_seconds
        data['bbox'] = self.bbox
        data['voxelsize'] = self.voxsize
        if video_filename:
            data['video_filename'] = video_filename
        if pictures_filename:
            data['pictures_filename'] = pictures_filename
        if extra_info:
            data['extra_info'] = extra_info

        """
        materials = { # original
            0: {'type': 'empty'},
            1: {'type': 'structural', 'stiffness':  100000, 'density': 1000, 'color': (255,0,0)}, # stiffness in MPa, density in Kg/m^3
            2: {'type': 'structural', 'stiffness':  500000, 'density': 1000, 'color': (0,255,0)},
            3: {'type': 'structural', 'stiffness': 1000000, 'density': 1000, 'color': (0,0,255)},
        }
        """
        """
        materials = { # softer
            0: {'type': 'empty'},
            1: {'type': 'structural', 'stiffness':   10000, 'density': 1000, 'color': (255,0,0)}, # stiffness in MPa, density in Kg/m^3
            2: {'type': 'structural', 'stiffness':   50000, 'density': 1000, 'color': (0,255,0)},
            3: {'type': 'structural', 'stiffness':  100000, 'density': 1000, 'color': (0,0,255)},
        }
        """
        materials = { # finer range
            0: {'type': 'empty'}
        }
        stiffness_range = np.linspace(10000, 100000, self.n_materials)
        red_range = np.linspace(255, 0, self.n_materials)
        blue_range = np.linspace(0, 255, self.n_materials)
        for i, (stiffness, red, blue) in enumerate(zip(stiffness_range, red_range, blue_range), start=1):
            materials[i] = {'type': 'structural', 'stiffness': stiffness, 'density': 1000, 'color': (red,0,blue)}

        pneumatics = {}
        for i, p in enumerate(self.pneu_indices):
            pneumatics[i] = {'quads': self.make_pneu_quads(p)}

        data['materials'] = materials
        data['voxels'] = self.voxels.tolist()
        data['pneumatics'] = pneumatics
        data['crosssection'] = 0.025 # Multiplier for flow rate, not really a physically-based quantity
        data['pressure'] = 1.2 # Air pressure in bar with which the pneumatic network is actuated

        """
        Controller architecture is as follows:
        first layer: 2 inputs, one sine and one cosine, of specified freq
        hidden layer: fully connected, n_hidden nodes, tanh activation
        output layer: n_out nodes, fully connected, tanh activation
        Both hidden and output layer have biases.
        Weights and biases range from +- 5 (so that the tanh can exhibit more
        step-like behaviour compared to just +- 1).
        """
        data['controller'] = self.genome.controller_config()

        """
        Possible values for 'control_type':
        'recurrent_closed': A modified Elman network, the context nodes have recurrent self-connections as well. The internal pressure of each chamber is made available to the controller as an input (in addition to the oscillators). The inlets and outlets of each chamber are controlled separately.
        'recurrent_open': Same as recurrent_open but without sensor inputs.
        """
        data['control_type'] = self.genome.control_type if hasattr(self.genome, 'control_type') else 'recurrent_open'

        data['terrain_type'] = self.genome.terrain_type if hasattr(self.genome, 'terrain_type') else 'flat'
        if self.genome.terrain_seed == -1:
            data['terrain_seed'] = random.randint(0, 2**16)
        else:
            data['terrain_seed'] = self.genome.terrain_seed

        data['directed_locomotion'] = self.genome.directed_locomotion if hasattr(self.genome, 'directed_locomotion') else False
        if hasattr(self.genome, 'evaluate_angle') and (self.genome.evaluate_angle is not None):
            assert data['directed_locomotion'], 'Only in directed locomotion experiments'
            data['evaluate_angle'] = self.genome.evaluate_angle

        with open(filename, 'w') as f:
            json.dump(data, f)
        return hash(json.dumps(data))

    def fitness(self, remove_tmp=True, eval_seconds_override=None, extra_info=None, video_filename=None, pictures_filename=None):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.join(curr_dir, '..')
        folder = os.path.join(root_dir, 'testbots')
        os.makedirs(folder, exist_ok=True)
        file_uuid = str(uuid.uuid1())
        fn = os.path.join(folder, f'{file_uuid}.json')
        if not remove_tmp:
            print(fn)

        eval_seconds = eval_seconds_override or self.genome.eval_seconds

        the_hash = self.export(fn, eval_seconds, video_filename, pictures_filename, extra_info)
        pneuvox = os.path.join(root_dir, 'pneuvox')
        proc_result = subprocess.run([pneuvox, fn], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        """
        os.makedirs('/var/scratch/tmk390/debuglogs/', exist_ok=True)
        with open(f'/var/scratch/tmk390/debuglogs/{file_uuid}.out', 'w') as f:
            f.write(proc_result.stdout.decode('utf-8'))
        with open(f'/var/scratch/tmk390/debuglogs/{file_uuid}.err', 'w') as f:
            f.write(proc_result.stderr.decode('utf-8'))
        """
        # print(proc_result.stdout.decode('utf-8'))
        # print(proc_result.stderr.decode('utf-8'))
        fitness = None
        fitness_distance = None
        if("void CVX_Link::updateForces(): Assertion `!(forceNeg.x != forceNeg.x) || !(forceNeg.y != forceNeg.y) || !(forceNeg.z != forceNeg.z)' failed" in proc_result.stderr.decode('utf-8')):
            # Catch a specific simulator error
            fitness = 0
            killed = 1
            secondsToRows = -1
            fitness_distance = 0
            print(f"Warning, killing robot due to CVX_Link failure: {fn}")
        else:
            for line in proc_result.stdout.decode('utf-8').splitlines()[::-1]:
                if line.startswith('Fitness:'):
                    fitness = float(line.split()[-1])
                    # print(f'Fitness: {fitness}')
                if line.startswith('Distance:'):
                    fitness_distance = float(line.split()[-1])
                if line.startswith('Killed:'):
                    killed = int(line.split()[-1])
                if line.startswith('var_Dpp0:'):
                    Dpp0 = float(line.split()[-1])
                if line.startswith('var_L:'):
                    L = float(line.split()[-1])
                if line.startswith('var_dB0B1:'):
                    dB0B1 = float(line.split()[-1])
                if line.startswith('var_Ppp1:'):
                    Ppp1 = float(line.split()[-1])
            if fitness is not None and (hasattr(self.genome, 'fused_fitness') and self.genome.fused_fitness) and not killed:
                fitness = {
                    'Dpp0': Dpp0,
                    'L': L,
                    'dB0B1': dB0B1,
                    'Ppp1': Ppp1
                }

        if fitness is None:
            # raise ValueError(f'No fitness from genome {fn}')
            # whatever:
            fitness = 0
            killed = 1
            secondsToRows = -1
            fitness_distance = 0
            print(f"Warning, killing robot due to unknown simulator problem: {fn}")
        else:
            if remove_tmp:
                os.remove(fn)
        if hasattr(self.genome, 'directed_locomotion') and self.genome.directed_locomotion:
            return fitness, fitness_distance, killed, the_hash
        else:
            return fitness, killed, the_hash
