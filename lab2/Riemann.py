import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


v_left = 0
v_right = 0

rho_left = 13
rho_right = 1.3

p_left = 10e5
p_right = 1e5

L = 10
tf = 0.02

gamma = 5/3
gamma_1 = gamma - 1
e_left = p_left/((gamma_1)*rho_left)
e_right = p_right/((gamma_1)*rho_right)

t = np.linspace(0, tf, 2001)
tau = np.diff(t)[0]
x_range = np.linspace(-L, L, 201)

print(t.shape, x_range.shape)

h = np.diff(x_range)[0]


m = len(x_range)//2
w = np.zeros([len(t)+1, len(x_range), 3])
w[0] = np.array([np.array([rho_left, rho_left*v_left, rho_left*e_left]) for x in x_range[:m]] 
                + [np.array([rho_right, rho_right*v_right, rho_right*e_right]) for x in x_range[m:]])

def calc_lambda(e,u):
    c = np.sqrt(gamma*gamma_1 * e)
    c = np.sqrt(gamma_1*gamma_1 * e)
    Lambda = np.diag(np.abs([u+c, u, u-c]))
    return Lambda

def func (w_prev, w_curr, w_next, A, omega, lambd_abs, omega_inv): 
   return w_curr - tau*(A @ (w_next-w_prev)/(2*h)) + tau*(omega_inv @ lambd_abs @ omega) @ (w_next-2*w_curr+w_prev)/(2*h)

for i in range(len(t)):
    for j in range(len(x_range)):
        # print(w[i][j])
        e = w[i][j][2]/w[i][j][0] #energy
        u = w[i][j][1]/w[i][j][0] #velocity
        c = np.sqrt(gamma*(gamma-1)*e) 

        lambda_abs = calc_lambda(e,u)
        A = np.array([[0, 1, 0], 
                      [-u**2, 2*u, gamma-1], 
                      [-gamma*u*e, gamma*e, u]])
        
        omega_t = np.array([[-u*c,c, gamma_1],
                         [-c*c,0,gamma_1],
                         [u*c, -c, gamma_1]])
        
        omega_t_inv = np.array([[1/(2*c**2), -1/c**2, 1/(2*c**2)], 
                               [(c + u)/(2*c**2), -u/c**2, (-c + u)/(2*c**2)], 
                               [1/(2*gamma_1), 0, 1/(2*gamma_1)]])

        CFL = np.max((lambda_abs)*tau/h)
        if CFL > 1:
            raise ValueError

        if (j == 0):
            w[i+1][j] = func(w[i][j], w[i][j+1], w[i][j+2], A, omega_t,lambda_abs,omega_t_inv)
        elif (j == len(x_range) - 1):
            w[i+1][j] = func(w[i][j-2], w[i][j-1], w[i][j], A, omega_t,lambda_abs,omega_t_inv)
        else:
            w[i+1][j] = func(w[i][j - 1], w[i][j], w[i][j+1], A, omega_t,lambda_abs,omega_t_inv)



def plotParams(time):
    fig, [[rhoax, uax], [eax, pax]] = plt.subplots(2, 2)

    n = int(len(t)*time/tf)
    rho = w[n, :, 0]
    rhoax.plot(x_range, rho)
    rhoax.set_xlabel(r'$x, м$')
    rhoax.set_ylabel(r'$\rho, кг/м^3$')
    rhoax.set_xlim(-10, 10)
    rhoax.grid()

    u = w[n, :, 1]/rho
    uax.plot(x_range, u)
    uax.set_xlabel(r'$x, м$')
    uax.set_ylabel(r'$u, м/c$')
    uax.set_xlim(-10, 10)
    uax.grid()

    e = w[n, :, 2]/rho
    e /= 1e3
    eax.plot(x_range, e)
    eax.set_xlabel(r'$x, м$')
    eax.set_ylabel(r'$e, кДж/кг$')
    eax.set_xlim(-10, 10)
    eax.grid()

    p = (gamma-1)*rho*e
    p /= 1e2
    pax.plot(x_range, p)
    pax.set_xlabel(r'$x, м$')
    pax.set_ylabel(r'$p, атм$')
    pax.set_xlim(-10, 10)
    pax.grid()

    fig.suptitle(f't={time}с, h={round(h, 2)}м, 'r'$\tau$'f'={tau}с')
    fig.set_figheight(10)
    fig.set_figwidth(16)
    fig.subplots_adjust(wspace=0.2)
    filename = f'plot_{int(time*1000):1d}.png'
    fig.savefig(filename)
    fig.clf()


times = np.linspace(0, 0.015, 200)
for time in times:
    plotParams(time)