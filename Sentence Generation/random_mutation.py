'''
Here, I will try to generate strings purely through mutation and survival.

The survival condition will be harsh i.e. if you aren't amongst the best, you
die.
'''
import numpy as np
import argparse
import string

parser = argparse.ArgumentParser()
parser.add_argument('--default',type=int,default=1)
parser.add_argument('--save_file',type=str,default='')
parser.add_argument('--var',type=float,default=0.5)
parser.add_argument('--population',type=int,default=10000)
parser.add_argument('--max_gen',type=int,default=10000)
parser.add_argument('--frequency',type=int,default=10)
parser.add_argument('--survival_rate',type=int,default=0.01)
parser.add_argument('--plot',type=int,default=1)

params = parser.parse_args()

np.random.seed(101)

vocab = ''.join(i+j for i,j in zip(string.ascii_uppercase,string.ascii_lowercase))
vocab = string.ascii_lowercase
vocab = list(vocab)
vocab.append(' ')

vocab_arr = np.array(vocab)

if params.default:
    target_str = 'a cat sat on a hat'
else:
    while True:
        target_str = input('Enter a sentence:')
        if all(x.isalpha() or x.isspace() for x in target_str):
            break
        print('Legal characters: English lower case and space.')

target_length = len(target_str)
target_arr = np.array(list(map(lambda x:vocab.index(x),target_str))).astype(np.int_)

# for plotting purposes
avg_fits = []
best_fits = []

#this measure sucks
if False:
    def fitness(arr):
        fitness.optimum = 1
        return np.sum(arr==target_arr) / target_length

# this measure rocks
def fitness(arr):
    temp = np.abs(arr-target_arr)
    return -np.linalg.norm( np.min( np.vstack((temp,len(vocab)-temp)), axis=0) )
fitness.optimum = 0


def mutate(sample):
    mutation = np.random.randn(*sample.shape) * np.sqrt(params.var)
    return (sample + mutation).astype(np.int_) % len(vocab)

def arr_to_str(arr):
    return ''.join(vocab_arr[arr])

def log_status(gNo,samples):
    avg_fitness = sum(map(fitness,samples)) / params.population
    fittest = max(samples, key=fitness)

    avg_fits.append(avg_fitness)
    best_fits.append(fitness(fittest))

    print('Generation {0}:'.format(gNo))
    print('Average fitness:{0:.2f}\t Best fitness:{1:.2f}'.format(avg_fitness,fitness(fittest)))
    print('Fittest sentence: ' + arr_to_str(fittest) + '\n\n')
    return fitness(fittest)

def decimate(samples):
    lst = sorted(samples,reverse=True,key=fitness)[:int(params.survival_rate * params.population)]
    return np.array(lst,dtype=np.int_)


def repopulate(old_gen):
    new_gen = [_ for _ in old_gen]
    while True:
        if len(new_gen) >= params.population:
            break
        new_gen.extend(mutate(old_gen))
    return np.array(new_gen[:params.population]).astype(np.int_)



# main simulation starts here
pop = (np.random.rand(params.population,target_length) * len(vocab)).astype(np.int_)

for gNo in range(params.max_gen + 1):
    if gNo % params.frequency == 0:
        fit_max = log_status(gNo,pop)
        if fit_max == fitness.optimum:
            print('Target accomplished:)')
            break
    survivors = decimate(pop)
    pop = repopulate(survivors)


if fit_max < fitness.optimum:
    print('Failed to reach optimal fitness:(')

if params.plot:
    import pylab as plt
    plt.figure()
    plt.title('Fitness over time')
    plt.xlabel('Generation count')
    plt.ylabel('Fitness')
    plt.plot(avg_fits,label='Average fitness')
    plt.plot(best_fits,label='Best fitness')
    plt.legend()
    plt.show()



exit()
