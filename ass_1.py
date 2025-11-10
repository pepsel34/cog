"""
Assignment 1 Processing Lab
"""

from itertools import product
import matplotlib.pyplot as plt

def start():
    time = 0
    return time

def perceptualstep(type='middle'):
    dict_time = {'slow': 200, 'middle': 100, 'fast': 50}
    time = dict_time[type]
    return time

def cognitivestep(type='middle'):
    dict_time = {'slow': 170, 'middle': 70, 'fast': 25}
    time = dict_time[type]
    return time

def motorstep(type='middle'):
    dict_time = {'slow': 100, 'middle': 70, 'fast': 30}
    time = dict_time[type]
    return time

def example1(type='middle'):
    start_time = start()
    perc_time = perceptualstep(type)
    cog_time = cognitivestep(type)
    motor_time = motorstep(type)
    total_time = start_time + perc_time + cog_time + motor_time 
    return total_time

def example2(completeness='extremes'):
    types_mode = ['slow', 'middle', 'fast']
    if completeness == "extremes":
        list_result = []
        for type in types_mode:
            list_result.append(example1(type))
    else:
        list_result = [
        start() + perceptualstep(p) + cognitivestep(c) + motorstep(m)
        for p, c, m in product(types_mode, repeat=3)
    ]
        plt.boxplot(list_result)
        plt.title("Distribution of Total Times (3×3×3 combinations)")
        plt.ylabel("Total Time (ms)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.xticks([])  # remove '1' on x-axis
        plt.ylim(bottom=0)  # set y-axis lower limit to 0
        plt.show()
    return list_result

def example3():
    

    pass

def main():
    total_time = example1()
    print(total_time)

    total_modes_time = example2('all')
    print(", ".join(str(item) for item in total_modes_time))

main()
