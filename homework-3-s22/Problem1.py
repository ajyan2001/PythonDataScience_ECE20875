import numpy as np
import matplotlib.pyplot as plt


def norm_histogram(hist):
    """
    takes a histogram of counts and creates a histogram of probabilities
    :param hist: a numpy ndarray object
    :return: list
    """
    # Create list
    list = []
    
    # Calculate and add probability to list
    for n in range(len(hist)):
        probability = hist[n] / sum(hist)
        list.append(probability)

    # Return list
    return list


def compute_j(histo, width):
    """
    takes histogram of counts, uses norm_histogram to convert to probabilties, it then calculates compute_j for one bin width
    :param histo: list
    :param width: float
    :return: float
    """
    # Get num samples
    m = sum(histo)
    # Convert histogram to probability
    histo = norm_histogram(histo)
    
    # Square probabilitys
    prob = [i ** 2 for i in histo]

    # Calculate j
    j = 2 / ((m - 1) * width) - (m + 1) / ((m - 1) * width) * sum(prob)
    # Return j
    return j



def sweep_n(data, minimum, maximum, min_bins, max_bins):
    """
    find the optimal bin
    calculate compute_j for a full sweep [min_bins to max_bins]
    please make sure max_bins is included in your sweep
    :param data: list
    :param minimum: int
    :param maximum: int
    :param min_bins: int
    :param max_bins: int
    :return: list
    """
    jlist = []
    # Convert data to list of counts
    for b in range(min_bins, max_bins + 1,1):
        n = plt.hist(data, bins = b, range = (minimum, maximum))[0]
        width = (maximum - minimum) / b
        j = compute_j(n, width) 
        
        jlist.append(j)

    return jlist


def find_min(l):
    """
    takes a list of numbers and returns the mean of the three smallest number in that list and their index.
    return as a tuple i.e. (the_mean_of_the_3_smallest_values,[list_of_the_3_smallest_values])
    For example:
        A list(l) is [14,27,15,49,23,41,147]
        The you should return ((14+15+23)/3,[0,2,4])

    :param l: list
    :return: tuple
    """
    list = l[:]
    list.sort()
    total = 0
    ind = []
    for n in range(0,3):
        total += list[n]
        mean = total / 3
        ind.append(l.index(list[n]))
    tup = (mean, ind)

    return tup

if __name__ == "__main__":
    data = np.loadtxt("input.txt")  # reads data from input.txt
    lo = min(data)
    hi = max(data)
    bin_l = 1
    bin_h = 100
    js = sweep_n(data, lo, hi, bin_l, bin_h)
    """
    the values bin_l and bin_h represent the lower and higher bound of the range of bins.
    They will change when we test your code and you should be mindful of that.
    """
    print(find_min(js))
