import numpy as np
import matplotlib.pyplot as plt

# first we work with a Levy distribution of step sizes
def levy_sampling(c, mu, num_steps):
    # we need to define pdf for which we need an almost-continuous range of x, so we use linspace with a short stepsize
    # as we need to choose a random sample of size num_steps later, to ensure randomness we define pdf for 100*num_steps   
    levy_x = np.linspace(mu + 0.01, mu + 10, num = 1000 * num_steps)
    levy_pdf = (c / np.sqrt(2 * np.pi)) * (np.exp(-c / (2 * (levy_x - mu))) / ((levy_x - mu)**(3/2)))
    
    step_lengths = np.random.choice(levy_x, size = num_steps, p = levy_pdf / np.sum(levy_pdf))
    step_directions = 2 * np.pi * np.random.sample(size = num_steps) # this returns random float between 0 and 2*pi
    
    # since we know the step length and direction (angle), we can use that to convert it into rectangular coordinates
    rect_step_xdir = step_lengths * np.cos(step_directions)
    rect_step_ydir = step_lengths * np.sin(step_directions)
    levy_rect_steps = np.array([rect_step_xdir, rect_step_ydir])
    
    # next we plot the distribution of stepsizes chosen above for future comparison
    plt.subplot(1, 3, 1)
    plt.hist(step_lengths, color = 'lightblue', ec = 'k')
    plt.title("Levy Distribution")
    
    levy_sampling.levy_median = np.median(step_lengths)
    print("Median of Levy sample:", levy_sampling.levy_median)
    
    return np.array(levy_rect_steps)


def gaussian_sampling(mu, sigma, num_steps):
    gaussian_x = np.linspace(mu, mu + 100 * sigma, 1000 * num_steps)
    gaussian_pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(gaussian_x - mu) ** 2 / (2 * sigma**2))
    
    step_lengths = np.random.choice(gaussian_x, size = num_steps, p = gaussian_pdf / np.sum(gaussian_pdf))
    step_directions = 2 * np.pi * np.random.sample(size = num_steps)
    
    rect_step_xdir = step_lengths * np.cos(step_directions)
    rect_step_ydir = step_lengths * np.sin(step_directions)
    gaussian_rect_steps = np.array([rect_step_xdir, rect_step_ydir])

    plt.subplot(1, 3, 2)
    plt.hist(step_lengths, color = 'lightblue', ec = 'k')
    plt.title("Gaussian Distribution")
    
    print("Median of Gaussian sample:", np.median(step_lengths))

    return np.array(gaussian_rect_steps)


def uniform_sampling(num_steps):
    step_lengths = np.random.uniform(0.0, 2 * levy_sampling.levy_median, num_steps)
    step_directions = 2 * np.pi * np.random.sample(size = num_steps)
    
    rect_step_xdir = step_lengths * np.cos(step_directions)
    rect_step_ydir = step_lengths * np.sin(step_directions)
    uniform_rect_steps = np.array([rect_step_xdir, rect_step_ydir])
    
    print("Median of Uniform sample:", np.median(step_lengths))
    
    plt.subplot(1, 3, 3)
    plt.hist(step_lengths, color = 'lightblue', ec = 'k')
    plt.title("Uniform Distribution")
    plt.suptitle("Step Lengths in Different Probability Distributions")
    plt.gcf().set_size_inches(15,4)
    plt.show()
    
    return np.array(uniform_rect_steps)


def move(rect_steps, num_steps):
    x_in = np.array([0, 0])
    x_list_in = np.array([0, 0])
    
    iterate = 1
    while iterate <= num_steps:
        x_in = np.add(x_in, rect_steps[:, iterate-1]) 
        iterate += 1
        x_list_in = np.append(arr = x_list_in, values = x_in)

    dim = int(x_list_in.size / 2)
    x_list = np.reshape(x_list_in, (dim, 2))

    return x_list

def plot_walk(levy_x_list, gaussian_x_list, uniform_x_list):
    # we access the x and y coordinates of each list of steps
    l_xpoints = levy_x_list[:, 0]
    l_ypoints = levy_x_list[:, 1]

    g_xpoints = gaussian_x_list[:, 0]
    g_ypoints = gaussian_x_list[:, 1]

    s_xpoints = uniform_x_list[:, 0]
    s_ypoints = uniform_x_list[:, 1]
    
    # plot the walks
    plt.plot(l_xpoints, l_ypoints, color = 'k', linewidth=0.65)
    plt.plot(g_xpoints, g_ypoints, linewidth=0.5)
    plt.plot(s_xpoints, s_ypoints, color='m', linewidth=0.5)
    
    plt.plot(0, 0, marker = 'o', color = 'r', ms = 11)
    plt.gcf().set_size_inches(10,10)
    plt.axis('equal')
    plt.grid()
    plt.show()    
        
def main():
    # parameters
    sigma = 1.9                             # standard deviation / scale parameter in Gaussian distribution
    c = 1                                   # scale paramter in Levy distribution
    mu = 0                                  # location parameter for the two distributions
    num_steps = 2000                        # number of steps to be taken
    
    # determine the steps of the walk 
    levy_rect_steps = levy_sampling(c, mu, num_steps)
    gaussian_rect_steps = gaussian_sampling(mu, sigma, num_steps)
    uniform_rect_steps = uniform_sampling(num_steps)
    
    # determine coordinates for moving
    levy_x_list = move(levy_rect_steps, int(num_steps))
    gaussian_x_list = move(gaussian_rect_steps, int(num_steps))
    uniform_x_list = move(uniform_rect_steps, int(num_steps))
    
    # plot the walk
    plot_walk(levy_x_list, gaussian_x_list, uniform_x_list)
        
if __name__ == "__main__":
    main()
