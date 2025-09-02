import numpy as np
import matplotlib.pyplot as plt

dataset = np.array([
    [15.5, 40, 0],
    [23.75, 23.25, 0],
    [8, 17, 1],
    [17, 21, 0],
    [5.5, 10, 1],
    [19, 12, 1],
    [24, 20, 0],
    [2.5, 12, 1],
    [7.5, 15, 1],
    [11, 26, 0]
])

# Extract features (age and temperature) and labels (pass/fail status)
features = dataset[:, :2]
labels = dataset[:, 2]

def objective_function(w):
    z=w[0]+w[1]*x_in
    z1=np.log(1+np.exp(-z))
    z2=np.log(1+np.exp(z))
    return np.dot(y_in,z1)+np.dot(y1_in,z2)

def gradient_function(w):
    hi=1/(1+np.exp(-w[0]-w[1]*x_in))
    yh=1.*hi-1.*y_in
    return np.array([np.sum(yh),np.dot(yh,x_in)])
    
def line_search(objective_function,gradient,x):
    beta=.1
    stepsize=1
    trial=100
    tau=.5
    for i in range(trial):
        fx1=objective_function(x)
        fx2=objective_function(x-stepsize*gradient)
        c=-beta*stepsize*np.dot(gradient,gradient)
        if fx2-fx1 <=c:
            break
        else:
            stepsize=tau*stepsize
    return stepsize
    
maxit=1000000;epsilon=1.e-3 

w=np.array([-2,3])
for i in range(maxit):
    gradient=gradient_function(w);b=np.linalg.norm(gradient)
    if b < epsilon:
        break
    stepsize=line_search(objective_function,gradient,w)
    w=w-stepsize * gradient
    print(i,b)

minimum_value = objective_function(w)

print("Minimum value:", minimum_value)
print("Minimum location:", w)
print("iteration:", i)

x_plot=np.linspace(2.5,24,100)
z=np.zeros(x_plot.size);p=np.zeros(x_plot.size)
z[:]=-w[0]-w[1]*x_plot[:]
p[:]=1/(1+np.exp(z[:]))
plt.scatter(x_in,y_in,color='black')
plt.plot(x_plot,p,color='blue')
plt.savefig('log_regression3.tif',dpi=1200)
plt.show()