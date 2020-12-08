#!/usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
import numpy
import sys

def read_file(filename):

    result_set = {}
    with open(filename) as results_file:

        for line in results_file.readlines():

            segments = line.split()

            order = int(segments[0])
            num_elem = int(segments[1])
            step_size = int(segments[2])
            
            data = [int(segments[i]) for i in [4, 5, 7, 8]]

            if order not in result_set: result_set[order] = {}
            if num_elem not in result_set[order]: result_set[order][num_elem] = {}
            if step_size not in result_set[order][num_elem]: result_set[order][num_elem][step_size] = []

            result_set[order][num_elem][step_size].append(data)


        #print('results:\n{}'.format(result_set))

    return result_set



def plot_result(result_set, order, num_elem):

    fig, ax = plt.subplots()

    if order not in result_set:
        print('AAAAhhh wrong order')
        return

    if num_elem not in result_set[order]:
        print('AAAhhh wrong num elem!')
        return

    for time_step, data in sorted(result_set[order][num_elem].items(), reverse = True):
        x = [i * time_step / 60 for i in range(1, len(data) + 1)]
        y_interp = [d[0] for d in data]
        y_no_interp = [d[2] for d in data]

        ax.plot(x, y_interp, '-*', label = 'dt = {}, interp'.format(time_step))
        ax.plot(x, y_no_interp,'-v', linestyle = 'dashed', label = 'dt = {}, no interp'.format(time_step))


    ax.grid()
    ax.set_xlabel('Time step (minutes)')
    ax.set_ylabel('Number of iterations during time step')
    plt.legend()

    plt.title('order = {}, #elements = {}'.format(order, num_elem))

    fig.savefig("test.png")
    plt.show()



if __name__ == '__main__':

    result_file = 'epi2out.txt'
    result_set = read_file(result_file)

    order = 5
    num_elem = 20

    if len(sys.argv) == 3:
        order = int(sys.argv[1])
        num_elem = int(sys.argv[2])


    plot_result(result_set, order, num_elem)

