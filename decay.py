import matplotlib.pyplot as plt
from matplotlib import style
#from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
style.use('ggplot')

lsta, lstb, lstc, lstd = [], [], [], []
a , b, c, d = 1, 1, 1, 1
decay1 = 0.998
decay2 = 0.997
decay3 = 0.996
decay4 = 0.992


for i in range(1000):
	a, b, c, d = a*decay1, b*decay2, c*decay3, d*decay4
	lsta.append(a)
	lstb.append(b)
	lstc.append(c)
	lstd.append(d)

plt.title('Decay curve', loc='center', weight='bold', color='Black')
plt.plot(lsta, color='blue')
plt.plot(lstb, color='orange')
plt.plot(lstc, color='green')
plt.plot(lstd, color='black')
plt.show()