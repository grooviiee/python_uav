from matplotlib import pyplot as plt
import numpy as np

x = np.arange(1,10,0.1)
y = x*0.2
y2 = np.sin(x)
plt.plot(x,y,'b',label='first')
plt.plot(x,y2,'r',label='second')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('matplotlib sample')
plt.legend(loc='upper right')
plt.show()



# draw multiple lines
days = [1,2,3]
az = [2,4,8]
pfizer = [5,1,3]
moderna = [1,2,5]

plt.plot(days, az)
plt.plot(days, pfizer)
plt.plot(days, moderna)

# draw rectangle