import numpy as np
import neat
from voxelbot import Voxelbot, VoxelbotGenome
import os
import uuid
from scipy.ndimage import binary_dilation
from pprint import pprint
import pickle
from math import sqrt
from reproduction import MIPReproduction

import random
seed = random.randint(0, 9999999999)
seed=6975974813
print("SEED:", seed)
random.seed(seed)

curr_dir = os.path.dirname(os.path.abspath(__file__))

def test_load_bot():
    config = neat.Config(VoxelbotGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             os.path.join(curr_dir, 'config_body.cfg'))

    p = neat.Population(config)
    genome = p.population[1]
    for _ in range(1000):
        genome.mutate(config.genome_config)
        bot = Voxelbot(genome, config)
        if not bot.valid:
            continue
        if bot.actuatable:
            break
        # mean = np.mean(bot.voxels.astype(bool))
        # if 0.1 < mean < 0.9:
        #     break
    bot = Voxelbot(genome, config)
    if bot.valid and bot.actuatable:
        bot.fitness()
    else:
        print("Invalid robot...")

def test_get_pneu_quads():
    config = neat.Config(VoxelbotGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             os.path.join(curr_dir, 'config_body.cfg'))

    p = neat.Population(config)
    genome = p.population[1]
    bot = Voxelbot(genome, config)
    bot.bbox = [4, 4, 4]
    pneu_voxels = np.zeros(bot.bbox)
    pneu_voxels[1:3,1:3,1:3] = 1
    wall_voxels = binary_dilation(pneu_voxels, structure=np.ones((3,3,3))).astype(int) - pneu_voxels.astype(bool)
    print(pneu_voxels)
    print(wall_voxels)
    for plane, direction in [
            # [(1, 1, 0), 1], # XY, positive Z
            # [(1, -1, 0), -1], # XY, negative Z
            # [(1, 0, 1), -1], # XZ, negative Y
            # [(1, 0, -1), 1], # XZ, positive Y
            [(0, 1, 1), 1], # YZ, positive X
            [(0, 1, -1), -1] # YZ, negative X
        ]:
        quads = bot.get_pneu_quads(wall_voxels, pneu_voxels, plane, direction)
        print(len(quads))
        pprint(quads)

def test_make_pneu_quads():
    config = neat.Config(VoxelbotGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             os.path.join(curr_dir, 'config_body.cfg'))

    p = neat.Population(config)
    genome = p.population[1]
    bot = Voxelbot(genome, config)
    bot.bbox = [4, 4, 4]
    voxels = np.ones(bot.bbox)
    voxels[1,1,1] = 3
    voxels[2,2,2] = 3
    bot.voxels = voxels
    bot.pneu_indices = [3]
    bot.fitness()

def test_serialize():
    config = neat.Config(VoxelbotGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             os.path.join(curr_dir, 'config_body.cfg'))

    p = neat.Population(config)
    genome = p.population[1]
    for _ in range(1000):
        genome.mutate(config.genome_config)
        bot = Voxelbot(genome, config)
        if not bot.valid:
            continue
        mean = np.mean(bot.voxels.astype(bool))
        if 0.1 < mean < 0.9:
            break
    bot = Voxelbot(genome, config)
    save_path = os.path.join(curr_dir, 'serializetestbots', '1.json')
    with open(save_path, 'wb') as f:
        pickle.dump(bot, f)
    with open(save_path, 'rb') as f:
        bot2 = pickle.load(f)
    assert(bot.fitness() == bot2.fitness())

def test_controller():
    config = neat.Config(VoxelbotGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             os.path.join(curr_dir, 'config_body.cfg'))

    p = neat.Population(config)
    genome = p.population[1]
    for _ in range(1000):
        genome.mutate(config.genome_config)
        bot = Voxelbot(genome, config)
        if not bot.valid:
            continue
        if bot.actuatable:
            break
    bot = Voxelbot(genome, config)
    if bot.valid and bot.actuatable:
        bot.fitness()
    else:
        print("Invalid robot...")

def test_sort_chambers():
    config = neat.Config(VoxelbotGenome, MIPReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             os.path.join(curr_dir, 'config_body.cfg'))

    p = neat.Population(config)
    genome = p.population[1]
    for _ in range(1000):
        genome.mutate(config.genome_config)
        bot = Voxelbot(genome, config)
        if not bot.valid:
            continue
        if len(bot.pneu_indices) >= 2:
            break
    bot = Voxelbot(genome, config)
    if bot.valid and bot.actuatable:
        bot.fitness()
    else:
        print("Invalid robot...")


class mock_cppn:
    def __init__(self, presence, pneu, material):
        self.presence = presence
        self.pneu = pneu
        self.material = material

    def activate(self, coords):
        (x, y, z) = coords
        x = int(x*14-0.5)
        y = int(y*14-0.5)
        z = int(z*10-0.5)
        return self.presence[x,y,z], self.pneu[x,y,z]*1.1, self.material[x,y,z], 0, 0


def test_twister():
    config = neat.Config(VoxelbotGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             os.path.join(curr_dir, 'config_body.cfg'))
    genome = VoxelbotGenome(1)
    genome.configure_new(config.genome_config)
    genome.bias_out = [20]*genome.n_out

    pres = np.zeros([14,14,10], dtype=int)
    pneu = np.zeros([14,14,10], dtype=int)
    """pneu[:7,:10,5] = np.array([
        [1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,0],
        [1,1,1,1,1,1,1,1,0,0],
        [1,1,1,1,1,0,0,0,0,0],
        [1,1,1,1,0,0,0,0,0,0]])
    pneu[:7,:10,4] = np.array([
        [1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,0],
        [1,1,1,1,1,1,1,1,1,0],
        [1,1,1,1,1,1,1,1,0,0],
        [1,1,1,1,1,0,0,0,0,0],
        [1,1,1,1,0,0,0,0,0,0]])
    pneu[:7,:10,3] = np.array([
        [1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,0],
        [1,1,1,1,1,1,1,1,1,0],
        [1,1,1,1,1,1,1,1,1,0],
        [1,1,1,1,1,1,1,1,0,0],
        [1,1,1,1,1,0,0,0,0,0],
        [1,1,1,1,0,0,0,0,0,0]])
    pneu[:7,:10,2] = np.array([
        [1,1,1,1,1,1,1,1,1,0],
        [1,1,1,1,1,1,1,1,1,0],
        [1,1,1,1,1,1,1,1,1,0],
        [1,1,1,1,1,1,1,1,1,0],
        [1,1,1,1,1,1,1,1,0,0],
        [1,1,1,1,1,0,0,0,0,0],
        [1,1,1,1,0,0,0,0,0,0]])
    pneu[:7,:10,1] = np.array([
        [1,1,1,1,1,1,1,1,1,0],
        [1,1,1,1,1,1,1,1,1,0],
        [1,1,1,1,1,1,1,0,0,0],
        [1,1,1,1,1,1,1,0,0,0],
        [1,1,1,1,1,1,0,0,0,0],
        [1,1,1,1,0,0,0,0,0,0],
        [1,1,0,0,0,0,0,0,0,0]])
    pneu[:7,:10,0] = np.array([
        [1,1,1,1,1,1,1,1,1,0],
        [1,1,1,1,1,1,1,1,1,0],
        [1,1,1,1,1,1,1,0,0,0],
        [1,1,1,1,1,1,0,0,0,0],
        [1,1,1,1,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0]])"""

    # odd ok, even not ok?
    # pneu[2:9,2:11,2:7]=1
    pneu[2:8,2:10,2:6]=1
    
    pres = binary_dilation(pneu, structure=np.ones((3,3,3))).astype(int)
    bot = Voxelbot(genome, config,
        debug_cppn=mock_cppn(pres, pneu, np.ones([14,14,10], dtype=int)))
    
    bot.fitness()


def test_fitness_penalty():
    penaltymult = 10

    fits = [
    {'Dpp0': 52.5612, 'L': 54.464, 'dB0B1': 0.0158233, 'Ppp1': 0.207026},
    {'Dpp0': 54.2591, 'L': 58.0826, 'dB0B1': 0.162896, 'Ppp1': 0.922561},
    {'Dpp0': 50.3431, 'L': 58.3999, 'dB0B1': 0.243054, 'Ppp1': 1.32796},
    {'Dpp0': 46.2303, 'L': 60.0011, 'dB0B1': 0.425914, 'Ppp1': 1.86681},
    {'Dpp0': 39.5979, 'L': 58.834, 'dB0B1': 0.555048, 'Ppp1': 2.07513}
    ]
    tot = 0
    for fit in fits:
        Dpp0 = fit['Dpp0']
        L = fit['L']
        dB0B1 = fit['dB0B1']
        Ppp1 = fit['Ppp1'] * penaltymult
        e = 0.00001
        part = abs(Dpp0) / (L + e) * (Dpp0 / (dB0B1 + 1) - Ppp1)
        print(f'partial: {part}')
        tot += part
    print(f'total 05 (bad): {tot}')

    fits = [
        {'Dpp0': 43.7565, 'L': 44.1879, 'dB0B1': 0.021374, 'Ppp1': 0.058299},
        {'Dpp0': 43.0816, 'L': 46.6144, 'dB0B1': 0.129733, 'Ppp1': 0.517935},
        {'Dpp0': 45.9887, 'L': 47.0661, 'dB0B1': 0.0654542, 'Ppp1': 0.255595},
        {'Dpp0': 46.9589, 'L': 48.2924, 'dB0B1': 0.0695794, 'Ppp1': 0.326001},
        {'Dpp0': 37.678, 'L': 42.1151, 'dB0B1': 0.0072955, 'Ppp1': 0.189288}
    ]
    tot = 0
    for fit in fits:
        Dpp0 = fit['Dpp0']
        L = fit['L']
        dB0B1 = fit['dB0B1']
        Ppp1 = fit['Ppp1'] * penaltymult
        e = 0.00001
        part = abs(Dpp0) / (L + e) * (Dpp0 / (dB0B1 + 1) - Ppp1)
        print(f'partial: {part}')
        tot += part
    print(f'total 01 (good): {tot}')


if __name__ == '__main__':
    # test_load_bot()
    # test_make_pneu_quads()
    # test_serialize()
    # test_controller()
    # test_sort_chambers()
    # test_twister()
    test_fitness_penalty()

