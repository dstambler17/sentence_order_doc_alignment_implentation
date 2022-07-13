from matplotlib import pyplot as plt
from modules.vector_modules.window_func import ModifiedPert

def plot_window_func(n):

    def convert_to_real(nums):
        res = []
        for num in nums:
            if type(num) == complex:
                res.append(0.001)
            else:
                res.append(num)
        return res

    '''
    Plots window funcs at diff j values
    '''

    steps = [i for i in range(0, n)]
    #steps[0] = 0.1 NOTE: IN THEIR PLOTS, FIRST

    mp = ModifiedPert(0, 16, n)
    zeros = convert_to_real([mp.evaluate_distribution(i, 0) for i in steps])
    twos = convert_to_real([mp.evaluate_distribution(i, 2) for i in range(0, n)])
    fives = convert_to_real([mp.evaluate_distribution(i, 5) for  i in range(0, n)])
    tens = convert_to_real([mp.evaluate_distribution(i, 10) for i in range(0, n)])
    fifteens = convert_to_real([mp.evaluate_distribution(i, 15) for i in range(0, n)])

    print(zeros)
    file_name ="window_plot"
    plt.plot(steps, zeros, label="0")
    plt.plot(steps, twos, label="2")
    plt.plot(steps, fives, label="5")
    plt.plot(steps, tens, label="10")
    plt.plot(steps, fifteens, label="15")
    plt.ylabel("Dist")
    plt.title("Window func")
    plt.legend(loc="lower right")
    plt.savefig('./devtools/plots/%s_.png' % file_name)
    plt.clf()

if __name__ == "__main__":
    plot_window_func(60)