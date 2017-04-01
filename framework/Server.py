# -*- coding: utf-8 -*-
import multiprocessing as mp
import os
import random
from signal import signal, SIGINT, SIG_IGN, siginterrupt
import time
import MLog
import JsonDataSource


MAX_Q_SIZE = 10240


def data_source():
    """
    Define A Simple Data Source Constructor
    :return: An Iterator
    """
    dataset = [0.1, 0.2, 0.3, 0.4, 0.5]
    while True:
        time.sleep(2)
        yield random.choice(dataset)


def proc_proxy(cntl_q, data_q, exit_flag):
    """
    worker process proxy
    :param cntl_q:
    :param data_q:
    :param exit_flag:
    :return:
    """
    for item in data_source():
        data_q.put(item)
        if exit_flag.is_set():
            cntl_q.put({'event': 'exit', 'pid': os.getpid()})
            break


def proc_worker(cntl_q, data_q):
    """
    worker process entry
    :param cntl_q:
    :param data_q:
    :return:
    """
    while True:
        item = data_q.get()
        handle(item)

    cntl_q.put({'event': 'exit', 'pid': os.getpid()})


def handle(item):
    """
    handle data item
    :param item:
    :return:
    """
    MLog.logger.debug('try handle data ' . format(item))


def run():
    """
    main process entry
    :return:
    """
    proc_pool = {}  # child process collection
    cntl_q = mp.Queue()  # control message queue
    data_q = mp.Queue(MAX_Q_SIZE)  # data message queue

    exit_flag = mp.Event()  # exit flag, default value FALSE

    signal(SIGINT, lambda x, y: exit_flag.set())

    siginterrupt(SIGINT, False)

    MLog.logger.info('main proc started:' . format(os.getpid()))
    proc = mp.Process(target=proc_proxy, args=(cntl_q, data_q, exit_flag))
    proc.start()
    proc_pool[proc.pid] = proc
    MLog.logger.info('proxy proc started:' . format(os.getpid()))

    for num in range(10):
        proc = mp.Process(target=proc_worker, args=(cntl_q, data_q))
        proc.start()
        proc_pool[proc.pid] = proc
        MLog.logger.info('worker proc started:'.format(os.getpid()))

    while True:
        item = cntl_q.get()
        if item['event'] == 'exit':
            proc = proc_pool.pop(item['pid'])
            proc.join()
            MLog.logger.info('worker proc exited:' . format(os.getpid()))
            proc = mp.Process(target=proc_worker, args=(cntl_q, data_q))
            proc.start()
            proc_pool[proc.pid] = proc
            MLog.logger.info('worker proc started:'.format(os.getpid()))
        else:
            MLog.logger.info('empty loop')

        if not proc_pool:  # all child-processes have been exited
            break

    MLog.logger.info('main() stopped' . format(os.getpid()))


if __name__ == '__main__':
    run()