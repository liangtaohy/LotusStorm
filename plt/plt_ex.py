import matplotlib.pyplot as plt
import numpy as np

plt.plot([1,2,3,4],[1,4,9,16], 'ro')
plt.ylabel("sample numbers")
plt.show()

plt.plot([1,2,3,4],[1,4,9,16], 'r--')
plt.ylabel("sample numbers")
# The axis() command in the example above takes a list of [xmin, xmax, ymin, ymax] and specifies the viewport of the axes.
plt.axis([0, 6, 0, 20])
plt.show()

# evenly sampled time at 200ms intervals
t = np.arange(0., 5., 0.2)

# red dashes, blue squares and green triangles
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()

def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

plt.figure(1)
plt.subplot(211)
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

plt.subplot(212)
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.show()