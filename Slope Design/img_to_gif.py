import imageio
from glob import glob
import os

filenames = glob(os.path.join('Images','*.png'))

with imageio.get_writer('movie.gif', mode='I', duration=0.2) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
        os.remove(filename)
