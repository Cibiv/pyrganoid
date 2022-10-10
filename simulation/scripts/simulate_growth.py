
from os.path import join as pjoin
import time
import sys



from simulations import *
from visualisation import *


if __name__ == '__main__':
    for a_rate in [0.7]:
        fs = [generate_model_san, generate_model_san2, generate_model_san3]
        titles = ['model_san', 'model_san_increased_s', 'model_san_increased_a']
        for f, title in zip(fs, titles):
            dats, out = f(a_rate)
            norm_dats = {day: 10 *  dats[day] / np.min(dats[day]) for day in dats}
            plot_all(norm_dats, f'results/{a_rate}/', title, font_size=10)

            if out:
                plot_celltype_composition(out, f'results/{a_rate}', title, font_size=10)
                plot_celltypes_ranked(11, dats, out, f'results/{a_rate}', title, font_size=10, plot_n=False)
                plot_celltypes_ranked(40, dats, out, f'results/{a_rate}', title, font_size=10, plot_n=False)


