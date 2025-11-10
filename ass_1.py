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
    results = []
    types_mode = ['slow', 'middle', 'fast']
    for p, c, m in product(types_mode, repeat=3):
        perc_tau = perceptualstep(p)
        cog_tau = cognitivestep(c) 
        mot_tau = motorstep(m)
        total = perc_tau * 2 + cog_tau * 2 + mot_tau

        results.append({
            'perc_tau': p,
            'cog_tau': c,
            'mot_tau': m,
            'total_time': total
        })

    return results

def example4():
    results = []
    types_mode = ['slow', 'middle', 'fast']
    second_stim_times = [40, 80, 110, 150, 210, 240] 
    for p, c, m in product(types_mode, repeat=3):
        perc_tau = perceptualstep(p)
        cog_tau = cognitivestep(c) 
        mot_tau = motorstep(m)

        for t in second_stim_times:
            start_second = max(t, perc_tau)
            end_second = start_second + perc_tau

            total = end_second + cog_tau * 2 + mot_tau

        results.append({
            'perc_tau': p,
            'cog_tau': c,
            'mot_tau': m,
            'total_time': total
        })

    return results

def example5(plot=True):
    types_mode = ['slow', 'middle', 'fast']
    results = []

    # base error 
    base_error = 0.01

    error_factor = {'slow': 0.5, 'middle': 2, 'fast': 3}

    for p, c, m in product(types_mode, repeat=3):
        tau_p = perceptualstep(p)
        tau_c = cognitivestep(c)
        tau_m = motorstep(m)

        total_time = 2 * tau_p + 2 * tau_c + tau_m

        error_prob = base_error
        error_prob *= error_factor[p] ** 2
        error_prob *= error_factor[c] ** 2
        error_prob *= error_factor[m]

        results.append({
            'perceptual_type': p,
            'cognitive_type': c,
            'motor_type': m,
            'total_time': total_time,
            'error_prob': error_prob
        })

    if plot:
        plt.figure(figsize=(8, 5))
        times = [r['total_time'] for r in results]
        errors = [r['error_prob'] for r in results]
        plt.scatter(times, errors, c='blue', alpha=0.7)
        plt.title("Example 5: Experiment time vs Error")
        plt.xlabel("Experiment time (ms)")
        plt.ylabel("Error (proportion)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

    return results


def main():
    total_time = example1()
    print(total_time)

    #total_modes_time = example2('all')
    #print(", ".join(str(item) for item in total_modes_time))

    # total_time_ex4 = example4()
    # print(total_time_ex4)

    total_time_ex5 = example5()
    print(total_time_ex5)


main()
