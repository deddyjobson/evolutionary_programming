'''
Here, I will try to generate strings purely through mutation and survival.
'''
import numpy as np
import argparse

from itertools import cycle

parser = argparse.ArgumentParser()
parser.add_argument('--default',type=int,default=1)
parser.add_argument('--save_file',type=str,default='')
parser.add_argument('--var',type=float,default=1)
parser.add_argument('--population',type=int,default=100)
parser.add_argument('--max_gen',type=int,default=1000)
parser.add_argument('--survival_rate',type=int,default=0.1)

params = parser.parse_args()

np.random.seed(101)

if params.default:
    target_str = 'I need some example sentence, so this will do.'
    target_str = 'This'
else:
    target_str = input('Enter a sentence:')

target_length = len(target_str)
target_arr = np.array(list(map(ord,target_str))).astype(np.uint8)


#this measure sucks
if False:
    def fitness(arr): 
        fitness.optimum = 1
        return np.sum(arr==target_arr) / target_length

# this measure rocks
def fitness(arr):
    fitness.optimum = 0
    return -np.linalg.norm(arr-target_arr)



def mutate(sample):
    mutation = np.random.randn(*sample.shape) * np.sqrt(params.var)
    return (sample + mutation).astype(np.uint8)

def arr_to_str(arr):
    return ''.join(list(map(chr,arr)))

def log_status(gNo,samples):
    avg_fitness = sum(map(fitness,samples)) / params.population
    fittest = max(samples, key=fitness)
    print('Generation {0}:'.format(gNo))
    print('Average fitness:{0:.2f}\t Best fitness:{1:.2f}'.format(avg_fitness,fitness(fittest)))
    print('Fittest sentence: ' + arr_to_str(fittest) + '\n\n')
    return fitness(fittest)

def decimate(samples):
    lst = sorted(samples,reverse=True,key=fitness)[:int(params.survival_rate * params.population)]
    return np.array(lst,dtype=np.uint8)

def repopulate(old_gen):
    new_gen = [_ for _ in old_gen]
    while True:
        if len(new_gen) >= params.population:
            break
        new_gen.extend(mutate(old_gen))
    return np.array(new_gen[:params.population]).astype(np.uint8)



# main simulation starts here
pop = (np.random.rand(params.population,target_length) * 256).astype(np.uint8)

for gNo in range(params.max_gen):
    fit_max = log_status(gNo,pop)
    if fit_max == fitness.optimum:
        print('Target accomplished:)')
        break
    survivors = decimate(pop)
    pop = repopulate(survivors)


if fit_max < fitness.optimum:
    print('Failed to reach optimal fitness:(')





exit()
