import numpy as np
import matplotlib.pyplot as plt

def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)
t3 = np.arange(0.0, 5.0, 0.1)
t4 = np.arange(0.0, 5.0, 0.02)

plt.figure(1).suptitle("Nono")
plt.subplot(221)
plt.plot(t1, f(t1), '+', t2, f(t2), 'k')

plt.subplot(222)
plt.plot(t2, np.cos(2*np.pi*t2), 'r')
plt.show()

plt.subplot(223)
plt.plot(t3, f(t3), 'bo', t3, f(t3), 'k')

plt.subplot(224)
plt.plot(t4, np.cos(2*np.pi*t4), 'r--')
plt.show()

