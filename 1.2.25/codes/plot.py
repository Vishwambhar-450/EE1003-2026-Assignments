import numpy as np
import matplotlib.pyplot as plt

Fs = 2000
T = 2
t = np.linspace(-T/2, T/2, Fs*T)

F = 50

x = np.sinc(2 * F * t)


def scale_signal(x, t, scale):
    t_scaled = t * scale
    return np.interp(t_scaled, t, x, left=0, right=0)

x_05 = scale_signal(x, t, 0.5)
x_1  = x
x_2  = scale_signal(x, t, 2)
y = x_05 + x_1 - x_2

def compute_fft(signal, Fs):
    N = len(signal)
    X = np.fft.fftshift(np.fft.fft(signal))
    f = np.fft.fftshift(np.fft.fftfreq(N, d=1/Fs))
    return f, np.abs(X)

f_x, X = compute_fft(x, Fs)
X = X / np.max(X)

plt.figure()
plt.plot(f_x, X)
plt.title("Spectrum of x(t) = sinc(2Ft)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")

plt.xlim(-150, 150)
plt.grid()
plt.show()

f, Y = compute_fft(y, Fs)

def estimate_bandwidth(f, spectrum, threshold=0.05):
    spectrum = spectrum / np.max(spectrum)

    indices = np.where(spectrum > threshold)[0]

    if len(indices) == 0:
        return 0

    max_freq = np.max(np.abs(f[indices]))
    return max_freq

bandwidth = estimate_bandwidth(f, Y)
nyquist_rate = 2 * bandwidth

print("Estimated Bandwidth:", bandwidth, "Hz")
print("Estimated Nyquist Rate:", nyquist_rate, "Hz")

plt.figure()
plt.plot(f, Y)
plt.title("Spectrum of y(t)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(-150, 150)
plt.grid()
plt.show()
