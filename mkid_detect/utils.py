import numpy as np

def remove_deadtime(t_list, dead_time=10):
    # TODO rewrite in C to speed up
    keep_times = []
    if len(t_list) == 0:
        return t_list

    for i, t in enumerate(t_list[:-1]):
        if (t_list[i+1] - t) > dead_time:
            keep_times.append(int(t))
        else:
            pass

    keep_times.append(int(t_list[-1]))

    return np.array(keep_times, dtype=int)