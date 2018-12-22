'''
The objective is to design a slope. The fitness of the slope here will be the
negative of the time taken for an object to slide down it.

I shall assume the object slides like a bead through string as in the entire
velocity keeps changing direction without energy loss
'''
import numpy as np
import argparse
import string
import png
import os

from skimage.draw import polygon


parser = argparse.ArgumentParser()
parser.add_argument('--var',type=float,default=1e-3) # enter normalized value
parser.add_argument('--population',type=int,default=1000)
parser.add_argument('--max_gen',type=int,default=10000)
parser.add_argument('--frequency',type=int,default=100)
parser.add_argument('--survival_rate',type=int,default=0.1)
parser.add_argument('--plot',type=int,default=0)

#slope parameters
parser.add_argument('--res',type=int,default=200)
parser.add_argument('--height',type=float,default=1)
parser.add_argument('--length',type=float,default=1)



params = parser.parse_args()

np.random.seed(101)
g = 9.8
params.var *= np.sqrt(params.height) # for unnormalizing purposes


# for plotting purposes
avg_fits = []
best_fits = []


def fitness(arr): # for a 1d array
    assert np.isclose( arr[-1],params.height) # safety check
    hs = arr[1:] - arr[:-1]
    if np.any(hs<=0):
        return -np.inf
    grad_inv = params.length / (hs * params.res)
    ts = np.sqrt(2*hs/g * (1+grad_inv**2))
    return -np.sum(ts)


def fitness(arr): # for a 1d array
    assert np.isclose(arr[-1],params.height) # safety check
    hs = arr[1:] - arr[:-1]
    vs = np.sqrt( 2*g*(params.height-arr + 1e-4) )
    # try:
    #     assert not np.isnan(vs).any()
    # except AssertionError:
    #     print(vs)
    #     print(vs-np.sort(vs))
    #     exit()
    grad_inv = params.length / (hs * params.res)
    sin_thetas = 1 / ( 1 + grad_inv**2 )
    ts = ( vs[:-1]-vs[1:] ) / ( g * sin_thetas )
    return -np.sum(ts)


def mutate(sample): # 1d and 2d
    mutation = np.random.randn(*sample.shape) * np.sqrt(params.var)
    temp = sample.copy()
    temp[...,1:] -= sample[...,:-1]
    temp = sample + mutation
    temp -= np.min(temp,axis=-1)[:,np.newaxis] - 0.01
    temp /= np.sum(temp, axis=-1)[:,np.newaxis] / params.height
    # try:
    #     assert np.all(temp>0)
    # except AssertionError:
    #     print(np.min(temp,axis=-1))
    #     print(temp)
    #     exit()
    return np.cumsum(temp,axis=-1)

def log_status(gNo,samples):
    avg_fitness = sum(map(fitness,samples)) / params.population
    fittest = max(samples, key=fitness)
    fittest_fitness = fitness(fittest)

    avg_fits.append(avg_fitness)
    best_fits.append(fittest_fitness)

    print('Generation {0}:'.format(gNo))
    print('Average fitness:{0:.2f}\t Best fitness:{1:.2f}'.format(avg_fitness,fittest_fitness))
    draw_slope(fittest,img_id=gNo)
    return fittest_fitness

def decimate(samples):
    lst = sorted(samples,reverse=True,key=fitness)[:int(params.survival_rate * params.population)]
    return np.array(lst)

def repopulate(old_gen):
    new_gen = [_ for _ in old_gen]
    while True:
        if len(new_gen) >= params.population:
            break
        new_gen.extend(mutate(old_gen))
    return np.array(new_gen[:params.population])

def draw_slope(arr,img_id='slope'):
    xx = []
    yy = []
    xs = np.linspace(0,params.length,params.res)
    ys = arr[::-1]

    # base
    xx.extend( [x for x in xs] )
    yy.extend( [0 for y in ys] )
    # top
    xx.extend( [x for x in xs[::-1]] )
    yy.extend( [y for y in ys[::-1]] )
    # side
    xx.extend( [0 for x in xs] )
    yy.extend( [y for y in ys] )

    xx = np.array(xx) * params.res / max(params.length,params.height)
    yy = np.array(yy) * params.res / max(params.length,params.height)


    img = np.zeros((params.res+1, params.res+1), dtype=np.uint8) + 255
    img[polygon(xx,yy)] = 0
    png.from_array(np.rot90(img),'L').save(os.path.join('Images','{0}.png'.format(img_id)))



# main simulation starts here
pop = np.random.rand(params.population,params.res)
pop /= np.sum(pop,axis=1)[:,np.newaxis]
pop *= params.height
pop = np.cumsum(pop,axis=1)


for gNo in range(params.max_gen + 1):
    if gNo % params.frequency == 0:
        fit_max = log_status(gNo,pop)
    survivors = decimate(pop)
    pop = repopulate(survivors)

os.system('python3 img_to_gif.py')

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
