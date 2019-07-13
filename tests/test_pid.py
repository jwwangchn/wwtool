import time
import numpy as np
import matplotlib.pyplot as plt

from wwtool import PID


if __name__ == '__main__':
    pid = PID(100.0, 0.0, 0.0)
    pid.setSampleTime(0.01)

    start_time = time.time()
    current = 0

    fig = plt.figure()
    plt.ion()
    for _ in range(1000):

        current_time = time.time()
        t = current_time - start_time

        target = np.sin(t) + np.random.rand() / 10.0 + np.cos(t / 3.0)
        
        pid.SetPoint = target
        pid.update(current)
        speed = pid.output

        current = current + speed * 0.01

        error = target - current

        plt.scatter(t, target, c='b', alpha=0.4, marker='*')
        plt.scatter(t, current, c='r', alpha=0.4)
        # plt.scatter(t, error, c='r', alpha=0.4)

        time.sleep(0.01)
        plt.pause(0.0001)

    plt.ioff()
    plt.show()