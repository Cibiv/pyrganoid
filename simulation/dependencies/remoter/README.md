Easy parallel processing

we want to exploratively execute algorithms in parallel on "big data" in python
When part of the program/algorithm changes, we don't want to calculate every previously calculated step again, but reuse the results that are reusable.

In explorative scientific analysis Datasets are often huge and the analysis often consists of many steps, where only the last one produces an output that can be evaluated by a human. This leads to two challanges: First computation time is often long due to big data size. Code changes in a step invalidate not only the results of the changed step, but also of all other steps.


```
import os
import time

def long_task(x):
    time.sleep(10)
    return x**2

from remoter import Server

if __name__ == "__main__":

    server = Server()

    with server as server:
        outs = [server.run(long_task, i) for i in range(10)]
        # start workers with python -m remoter.remoter 

    outs = [o.result for o in outs]
    print(outs)

    out_3 = server.run(long_task, 3)
    # no workers necessary - task is cached in memory by default
    print(out_3.result)
```
