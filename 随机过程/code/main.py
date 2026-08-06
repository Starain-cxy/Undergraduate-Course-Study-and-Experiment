import numpy as np
import random
import matplotlib.pyplot as plt

def CutbyTime(time,lambd):
    t=random.expovariate(lambd)
    series=[]
    while t<time:
        series.append(t)
        random_number = random.expovariate(lambd)
        t=t+random_number
    event= len(series)
    return series,time,event

def CutbyEvent(Event,lambd):
    n = 0
    t=0
    series = []
    while n<Event:
        random_number = random.expovariate(lambd)
        n=n+1
        t = t + random_number
        series.append(t)
    return series,t,Event

def plot_poisson_process(series, t, event,title):

    # x和y坐标
    times = [0] + series  # 从时间0开始
    counts = list(range(event + 1))  # 0, 1, 2, ..., event

    plt.step(times, counts, where='post')
    plt.xlabel('Time')
    plt.ylabel('Number of Events')
    plt.title(title)
    plt.grid(True)
    plt.show()

def main():
    series1, t1, event1=CutbyTime(150,0.1)
    plot_poisson_process(series1, t1, event1,"Possion Process(λ=0.1) Cut by Time=150")
    series2, t2, event2 = CutbyEvent(20, 0.1)
    plot_poisson_process(series2, t2, event2, "Possion Process(λ=0.1) Cut by N=20")
    print(f"按时间截断模拟事件发生的时间：{series1}")
    print(f"按时间截断模拟事件发生的时间：{series2}")

if __name__ == "__main__":
    main()
