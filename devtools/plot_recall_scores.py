import argparse
from matplotlib import pyplot as plt

def plot_window_func(scores, file_name):
    steps = [1, 3, 8]
    
    assert len(scores) == len(steps)

    plt.plot(steps, scores, label="Main method")
    plt.ylabel("Recall @ K")
    plt.title("Recall Plot")
    plt.legend(loc="lower right")
    plt.savefig('./devtools/plots/recall_plot_%s_.png' % file_name)
    plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--recall_scores', type=int, nargs='+',
                    help='recall scores')
    parser.add_argument('--name', help="file name for output")

    args = parser.parse_args()
    plot_window_func(args.recall_scores, args.name)