#Assignment 1 section 1
import matplotlib.pyplot as plt
from itertools import product

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
    cogn_time = cognitivestep(type)
    motor_time = motorstep(type)
    total_time = start_time + perc_time + cogn_time + motor_time
    return total_time

def example2(completedness='extremes'):
    type_list = ['fast', 'middle', 'slow']
    type_result = []
    if completedness == 'extremes':
        for type in type_list:
            total_time = example1(type)
            type_result.append(total_time)
    else:
        for (p, c, m) in product(type_list, repeat=3):  #more efficient way to loop 3 times
            total_time = start() + perceptualstep(p) + cognitivestep(c) + motorstep(m)
            type_result.append(total_time)
        plt.boxplot(type_result)
        plt.ylabel('time (ms)')
        plt.title('Distribution of all outcomes')
        plt.ylim(bottom=0)
        plt.show()
    return type_result

#print(example2('all'))

def example3():
    type_result = []
    type_list = ['fast', 'middle', 'slow']
    for (p, c, m) in product(type_list, repeat=3):
        perc_time = perceptualstep(p)
        cogn_time = cognitivestep(c)
        motor_time = motorstep(m)
        total_time = start() + 2*perc_time + 2*cogn_time + motor_time

        type_result.append({
            'perc_tau': p,
            'cogn_tau': c,
            'motor_tau': m,
            'total_time': total_time
        })
    return type_result

#print(example3())

def example4():
    type_result = []
    type_list = ['fast', 'middle', 'slow']
    stim2_timing = [40, 80, 110, 150, 210, 240]
    for (p, c, m) in product(type_list, repeat=3):
        perc_time = perceptualstep(p)
        cogn_time = cognitivestep(c)
        motor_time = motorstep(m)
        #copied code from Pepijn
        for stim2_time in stim2_timing:
            start_second = max(stim2_time, perc_time)   #checks which is larger to see what the start time of stim2 is
            end_second = start_second + perc_time

            total_time = end_second + cogn_time * 2 + motor_time

        type_result.append({
            'stim2_time' : stim2_time,
            'perc_tau': p,
            'cogn_tau': c,
            'motor_tau': m,
            'total_time': total_time
        })
    return type_result

#print(example4())

def example5():
    base_error = 0.01
    error_factor = {'slow': 0.5, 'middle': 2, 'fast': 3}
    type_result = []
    type_list = ['fast', 'middle', 'slow']
    for (p, c, m) in product(type_list, repeat=3):
        perc_time = perceptualstep(p)
        cogn_time = cognitivestep(c)
        motor_time = motorstep(m)
        total_time = start() + 2*perc_time + 2*cogn_time + motor_time

        error_prob = base_error
        error_prob *= error_factor[p] * error_factor[p]
        error_prob *= error_factor[c] * error_factor[c]
        error_prob *= error_factor[m]

        type_result.append({
            'perc_tau': p,
            'cogn_tau': c,
            'motor_tau': m,
            'total_time': total_time,
            'error_prob': error_prob
        })
    plt.figure(figsize=(8,6))
    times = [r['total_time'] for r in type_result]
    errors = [r['error_prob'] for r in type_result]
    plt.xlabel('Time (ms)')
    plt.ylabel('Error Probability')
    plt.scatter(times, errors)
    plt.show()
    return type_result

# print(example5())

#Section 2
