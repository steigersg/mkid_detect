import numpy as np

def remove_deadtime(t_list, dead_time=10):
    # TODO rewrite in C to speed up
    keep_times = []

    for i, t in enumerate(t_list[:-1]):
        if t_list[i+1] - t > dead_time:
            keep_times.append(int(t))
        else:
            pass

    return keep_times