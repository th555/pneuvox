import os
import sys
import pickle
import gzip
import random

from run import restore_checkpoint, SaveResultReporter
from voxelbot import Voxelbot

def replaybot(filename):
    with open(filename, 'rb') as f:
        bot = pickle.load(f)

    # bot.config.genome_config.terrain_type = 'rows'
    # bot.genome.terrain_type = 'rows'
    bot.genome.evaluate_angle = None
    bot.genome.fused_fitness = False

    print(bot.genome.freq[0])

    # Uncomment to reconstruct the bot based on genome
    botx = Voxelbot(bot.genome, bot.config)
    if type(botx.genome.freq) == float:
        botx.genome.freq = [botx.genome.freq]
    if not hasattr(bot.genome, 'terrain_seed'):
        bot.genome.terrain_seed = 100
    # bot.genome.terrain_type = 'rows'

    extra_info = []
    expname = os.path.split(filename)[1]
    expname = expname.replace('div20_1ppl', 'div20')
    splt = expname.split('_')
    if splt[0] == 'div20':
        extra_info.append("gait learning")
    elif splt[0] == 'mdv2':
        extra_info.append("directed locomotion")
    if splt[1] == 'open':
        extra_info.append("open-loop control")
    elif splt[1] == 'closed':
        extra_info.append("closed-loop control")
    try:
        if splt[2]:
            extra_info.append(f"experiment #{int(splt[2]):02}")
    except ValueError:
        pass

    results = botx.fitness(remove_tmp=False,
        eval_seconds_override=9999,
        extra_info=extra_info,
        video_filename=filename.split('/')[-1]
        # pictures_filename=filename.split('/')[-1]
    )
    """ to join the pictures into a timelapse:
    convert -alpha off +append screenshots/*.png screenshots/out.png
    """
    print(f'Fitness: {results[0]}')

def maketimelapse(filename):
    with open(filename, 'rb') as f:
        bot = pickle.load(f)
# Uncomment to reconstruct the bot based on genome

    # bot.config.genome_config.terrain_type = 'rows'
    # bot.genome.terrain_type = 'rows'
    bot.genome.evaluate_angle = None
    bot.genome.fused_fitness = False

    print(bot.genome.freq[0])

    botx = Voxelbot(bot.genome, bot.config)
    if type(botx.genome.freq) == float:
        botx.genome.freq = [botx.genome.freq]
    if not hasattr(bot.genome, 'terrain_seed'):
        bot.genome.terrain_seed = 100
    # bot.genome.terrain_type = 'rows'
    fn = filename.split('/')[-1]
    results = botx.fitness(remove_tmp=False,
        eval_seconds_override=9999,
        pictures_filename=fn
    )
    """ to join the pictures into a timelapse:
    convert -alpha off +append screenshots/*.png screenshots/out.png
    """
    os.system(f"convert -alpha off +append screenshots/{fn}*.png screenshots/timelapse_{fn}.png")
    os.system(f"rm screenshots/{fn}*.png")
    print(f'Fitness: {results[0]}')

def maketimelapsesteering(filename):
    fn = filename.split('/')[-1]
    for angle, anglename in [(40, 'left'), (-40, 'right')]:
        with open(filename, 'rb') as f:
            bot = pickle.load(f)

        # bot.config.genome_config.terrain_type = 'rows'
        # bot.genome.terrain_type = 'rows'
        bot.genome.evaluate_angle = angle
        bot.genome.fused_fitness = False

        print(bot.genome.freq[0])

        botx = Voxelbot(bot.genome, bot.config)
        if type(botx.genome.freq) == float:
            botx.genome.freq = [botx.genome.freq]
        if not hasattr(bot.genome, 'terrain_seed'):
            bot.genome.terrain_seed = 100
        # bot.genome.terrain_type = 'rows'
        results = botx.fitness(remove_tmp=False,
            eval_seconds_override=9999,
            pictures_filename=fn
        )
        """ to join the pictures into a timelapse:
        convert -alpha off +append screenshots/*.png screenshots/out.png
        """
        os.system(f"convert -alpha off +append screenshots/{fn}*.png screenshots/timelapse_{fn}_{anglename}.png")
        os.system(f"rm screenshots/{fn}*.png")
    os.system(f"convert -alpha off -append screenshots/timelapse_{fn}_left.png screenshots/timelapse_{fn}_right.png screenshots/timelapse_leftright_{fn}.png")
    os.system(f"rm screenshots/timelapse_{fn}_left.png screenshots/timelapse_{fn}_right.png")
    print(f'Fitness: {results[0]}')





def dbg(pop):
    genome = pop.population[37445]
    bot = Voxelbot(genome, pop.config)
    pop.config.genome_config.weight_mutate_power = 0.005
    pop.config.genome_config.bias_mutate_power = 0.005
    # import pdb; pdb.set_trace()
    if bot.valid: bot.fitness(remove_tmp=False, eval_seconds_override=9999)
    while 1:
        child = pop.config.genome_type(123)
        child.configure_crossover(genome, genome, pop.config.genome_config, cross_morphology=True, cross_controller=True)
        child.mutate(bot.config.genome_config, mutate_morphology=True, mutate_controller=True)
        bot2 = Voxelbot(child, pop.config)
        if bot2.valid: bot2.fitness(remove_tmp=False, eval_seconds_override=9999)

def replaycheckpoint(filename):
    pop = restore_checkpoint(filename)
    # dbg(pop); return
    for key, genome in pop.population.items():
        print(key)
        bot = Voxelbot(genome, pop.config)
        if bot.valid:
            results = bot.fitness(remove_tmp=False,
                eval_seconds_override=9999,
                # video_filename=filename.split('/')[-1]
            )

if __name__ == '__main__':
    path = sys.argv[1]
    if os.path.isfile(path):
        if 'checkpoint' in path.split('/')[-1]:
            replaycheckpoint(path)
        else:
            replaybot(path)
    elif os.path.isdir(path):
        for file in sorted(os.listdir(path)):
            print(file)
            replaybot(os.path.join(path, file))

