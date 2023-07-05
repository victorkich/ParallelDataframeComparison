from statsmodels.stats.power import TTestIndPower
import scipy.stats as st
from scipy.stats import iqr
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import matplotlib
import numpy as np
import pandas
import time


def main():
    with open('pilot_study.npy', 'rb') as f:
        data = np.load(f)
        data = [array.tolist() for array in data]
        data = np.array(data)

    datasets = ["circles.csv",  "tripadvisor.csv", "taxi.csv"]

    data_pandas_read = [data[0, :, 0], data[1, :, 0], data[2, :, 0]]
    data_modin_read = [data[0, :, 1], data[1, :, 1], data[2, :, 1]]

    matplotlib.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots()
    fig.set_size_inches(11, 8)

    pandas_confidence_low = []
    pandas_confidence_high = []
    pandas_mean = []
    modin_confidence_low = []
    modin_confidence_high = []
    modin_mean = []

    for i in range(3):
        # Pandas
        confidence_lower, confidence_higher = st.t.interval(confidence=0.983, df=len(data_pandas_read[i])-1, loc=np.mean(data_pandas_read[i]), scale=st.sem(data_pandas_read[i]))
        print("Mean pandas:", np.mean(data_pandas_read[i]), "{} < p < {}".format(confidence_lower, confidence_higher), "Std:", np.std(data_pandas_read[i]))
        pandas_confidence_low.append(confidence_lower)
        pandas_confidence_high.append(confidence_higher)
        pandas_mean.append(np.mean(data_pandas_read[i]))
        
        # Modin
        confidence_lower, confidence_higher = st.t.interval(confidence=0.983, df=len(data_modin_read[i])-1, loc=np.mean(data_modin_read[i]), scale=st.sem(data_modin_read[i]))
        print("Mean modin:", np.mean(data_modin_read[i]), "{} < p < {}".format(confidence_lower, confidence_higher), "Std:", np.std(data_modin_read[i]))
        modin_confidence_low.append(confidence_lower)
        modin_confidence_high.append(confidence_higher)
        modin_mean.append(np.mean(data_modin_read[i]))
        
    ax.plot(["11.8kb", "60.6mb", "209.4mb"], pandas_mean, linestyle='-', linewidth=2, label="Sequential", c="firebrick")
    ax.fill_between(["11.8kb", "60.6mb", "209.4mb"], pandas_confidence_low, pandas_confidence_high, alpha=0.1, facecolor="firebrick")
    ax.plot(["11.8kb", "60.6mb", "209.4mb"], modin_mean, linestyle='-', linewidth=2, label="Parallel", c="deepskyblue")
    ax.fill_between(["11.8kb", "60.6mb", "209.4mb"], modin_confidence_low, modin_confidence_high, alpha=0.1, facecolor="deepskyblue")
    ax.legend(loc="lower left", mode="expand", ncol=3, prop={'size': 20})
    ax.set_xlabel('CSV size', fontsize=20)
    ax.set_ylabel('Time (s)', fontsize=20)
    ax.set_title('Read CSV Spended Time - Pilot study (n=10)', fontsize=24)
    ax.set_ylim([-1, 7])
    ax.grid()
    fig.set_dpi(100)
    plt.savefig("read_csv_pilot.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    # ----------------------------------------------------------------------------------------------------------------

    sample_1 = data_pandas_read
    sample_2 = data_modin_read

    n1 = len(sample_1)
    n2 = len(sample_2)

    print(np.mean(sample_1))
    print(np.mean(sample_2))

    # calculate standard deviation for each sample
    std_dev_1 = np.std(sample_1, ddof=1)  
    std_dev_2 = np.std(sample_2, ddof=1)

    # calculate pooled standard deviation
    pooled_std_dev = np.sqrt(((n1 - 1)*std_dev_1**2 + (n2 - 1)*std_dev_2**2) / (n1 + n2 - 2))
    print('Pooled standard deviation: ', pooled_std_dev)

    # ----------------------------------------------------------------------------------------------------------------

    delta = 1
    sd = pooled_std_dev

    # parameters for the analysis 
    effect_size = delta / sd  # ratio of difference to standard deviation, delta is the minimum difference of interest
    alpha = 1 - 0.983  # significance level
    power = 0.95   # desired power

    analysis = TTestIndPower()
    sample_size = analysis.solve_power(effect_size = effect_size, power = power, alpha = alpha)

    print(sample_size)

    # -----------------------------------------------------------------------------------------------------------------

    with open('data.npy', 'rb') as f:
        data = np.load(f)
        data = [array.tolist() for array in data]
        data = np.array(data)

    datasets = ["circles.csv",  "tripadvisor.csv", "taxi.csv"]

    data_pandas_read = [data[0, :, 0], data[1, :, 0], data[2, :, 0]]
    data_modin_read = [data[0, :, 1], data[1, :, 1], data[2, :, 1]]

    matplotlib.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots()
    fig.set_size_inches(11, 8)

    pandas_confidence_low = []
    pandas_confidence_high = []
    pandas_mean = []
    modin_confidence_low = []
    modin_confidence_high = []
    modin_mean = []

    for i in range(3):
        # Pandas
        confidence_lower, confidence_higher = st.t.interval(confidence=0.983, df=len(data_pandas_read[i][:84])-1, loc=np.mean(data_pandas_read[i][:84]), scale=st.sem(data_pandas_read[i][:84]))
        print("Mean pandas:", np.mean(data_pandas_read[i][:84]), "{} < p < {}".format(confidence_lower, confidence_higher), "Std:", np.std(data_pandas_read[i][:84]))
        pandas_confidence_low.append(confidence_lower)
        pandas_confidence_high.append(confidence_higher)
        pandas_mean.append(np.mean(data_pandas_read[i][:84]))
        
        # Modin
        confidence_lower, confidence_higher = st.t.interval(confidence=0.983, df=len(data_modin_read[i][:84])-1, loc=np.mean(data_modin_read[i][:84]), scale=st.sem(data_modin_read[i][:84]))
        print("Mean modin:", np.mean(data_modin_read[i][:84]), "{} < p < {}".format(confidence_lower, confidence_higher), "Std:", np.std(data_modin_read[i][:84]))
        modin_confidence_low.append(confidence_lower)
        modin_confidence_high.append(confidence_higher)
        modin_mean.append(np.mean(data_modin_read[i][:84]))
        
    ax.plot(["11.8kb", "60.6mb", "209.4mb"], pandas_mean, linestyle='-', linewidth=2, label="Sequential", c="firebrick")
    ax.fill_between(["11.8kb", "60.6mb", "209.4mb"], pandas_confidence_low, pandas_confidence_high, alpha=0.1, facecolor="firebrick")
    ax.plot(["11.8kb", "60.6mb", "209.4mb"], modin_mean, linestyle='-', linewidth=2, label="Parallel", c="deepskyblue")
    ax.fill_between(["11.8kb", "60.6mb", "209.4mb"], modin_confidence_low, modin_confidence_high, alpha=0.1, facecolor="deepskyblue")
    ax.legend(loc="lower left", mode="expand", ncol=3, prop={'size': 20})
    ax.set_xlabel('CSV size', fontsize=20)
    ax.set_ylabel('Time (s)', fontsize=20)
    ax.set_title('Read CSV - Spended Time', fontsize=24)
    ax.set_ylim([-1, 7])
    ax.grid()
    fig.set_dpi(100)
    plt.savefig("read_csv.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    # -----------------------------------------------------------------------------------------------------------------

    print("CSV size: 11.8kb")

    sample_1 = data_pandas_read[0][:84]
    sample_2 = data_modin_read[0][:84]

    box_plot = plt.boxplot([sample_1, sample_2], vert=True, patch_artist=True, labels=['Sequential', 'Parallel'])

    colors = ["firebrick", "deepskyblue"]
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)

    plt.title('Box plot of Spended Time - 11.8kb')
    plt.ylabel('Read CSV - Spended Time')
    plt.xlabel('Computing Type')
    plt.savefig("read_csv_boxplot_1.pdf", format='pdf', bbox_inches="tight")
    plt.show()

    # -----------------------------------------------------------------------------------------------------------------

    # Calculate 50th percentile (median)
    median_1 = np.percentile(sample_1, 50)
    median_2 = np.percentile(sample_2, 50)

    print(f"50th percentile for sample 1: {median_1}")
    print(f"50th percentile for sample 2: {median_2}")

    # Identify outliers
    # Calculate the IQR for each sample
    iqr_1 = iqr(sample_1)
    iqr_2 = iqr(sample_2)

    # Define upper and lower bounds for outliers
    upper_bound_1 = np.percentile(sample_1, 75) + 1.5 * iqr_1
    lower_bound_1 = np.percentile(sample_1, 25) - 1.5 * iqr_1
    upper_bound_2 = np.percentile(sample_2, 75) + 1.5 * iqr_2
    lower_bound_2 = np.percentile(sample_2, 25) - 1.5 * iqr_2

    # Identify outliers
    outliers_1 = [x for x in sample_1 if x < lower_bound_1 or x > upper_bound_1]
    outliers_2 = [x for x in sample_2 if x < lower_bound_2 or x > upper_bound_2]

    print(f"Number of outliers in sample 1: {len(outliers_1)}")
    print(f"Number of outliers in sample 2: {len(outliers_2)}")

    # -----------------------------------------------------------------------------------------------------------------

    # Set the bins to be from 0 to 1900 with step size 100
    bins = np.arange(0, 0.35, 0.05)

    # Plot histogram for sample_1 in blue color
    plt.hist(sample_1, bins, alpha=0.5, label='Sequential', color="firebrick")

    # Plot histogram for sample_2 in green color
    plt.hist(sample_2, bins, alpha=0.5, label='Parallel', color="deepskyblue")

    # Add legend
    plt.legend(loc='upper right')

    plt.title('Read CSV - Spended Time Distribution - 11.8kb')

    # Label the axes
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency')

    plt.savefig("read_csv_distribution_1.pdf", format='pdf', bbox_inches="tight")
    # Show the plot
    plt.show()

    # -----------------------------------------------------------------------------------------------------------------

    t_stat, p_value = ttest_ind(sample_1, sample_2)

    print('t-statistic:', t_stat)
    print('p-value:', p_value)

    # -----------------------------------------------------------------------------------------------------------------

    # let's assume sample_1 and sample_2 are your samples
    std_dev_1 = np.std(sample_1, ddof=1)
    std_dev_2 = np.std(sample_2, ddof=1)

    n1 = len(sample_1)
    n2 = len(sample_2)

    pooled_std_dev = np.sqrt(((n1 - 1) * std_dev_1**2 + (n2 - 1) * std_dev_2**2) / (n1 + n2 - 2))
    delta = 1
    sample_size = 84

    effect_size = delta / pooled_std_dev  # ratio of difference to standard deviation, delta is the minimum difference of interest
    alpha = 1 - 0.983  # significance level
    nobs = sample_size  # number of observations

    analysis = TTestIndPower()
    power = analysis.solve_power(effect_size = effect_size, nobs1 = nobs, alpha = alpha)

    print("Power of the test:", power, "Effect size:", effect_size)

    # =================================================================================================================

    print("CSV size: 60.6mb")

    sample_1 = data_pandas_read[1][:84]
    sample_2 = data_modin_read[1][:84]

    box_plot = plt.boxplot([sample_1, sample_2], vert=True, patch_artist=True, labels=['Sequential', 'Parallel'])

    colors = ["firebrick", "deepskyblue"]
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)

    plt.title('Box plot of Spended Time - 60.6mb')
    plt.ylabel('Read CSV - Spended Time')
    plt.xlabel('Computing Type')
    plt.savefig("read_csv_boxplot_2.pdf", format='pdf', bbox_inches="tight")
    plt.show()

    # -----------------------------------------------------------------------------------------------------------------

    # Calculate 50th percentile (median)
    median_1 = np.percentile(sample_1, 50)
    median_2 = np.percentile(sample_2, 50)

    print(f"50th percentile for sample 1: {median_1}")
    print(f"50th percentile for sample 2: {median_2}")

    # Identify outliers
    # Calculate the IQR for each sample
    iqr_1 = iqr(sample_1)
    iqr_2 = iqr(sample_2)

    # Define upper and lower bounds for outliers
    upper_bound_1 = np.percentile(sample_1, 75) + 1.5 * iqr_1
    lower_bound_1 = np.percentile(sample_1, 25) - 1.5 * iqr_1
    upper_bound_2 = np.percentile(sample_2, 75) + 1.5 * iqr_2
    lower_bound_2 = np.percentile(sample_2, 25) - 1.5 * iqr_2

    # Identify outliers
    outliers_1 = [x for x in sample_1 if x < lower_bound_1 or x > upper_bound_1]
    outliers_2 = [x for x in sample_2 if x < lower_bound_2 or x > upper_bound_2]

    print(f"Number of outliers in sample 1: {len(outliers_1)}")
    print(f"Number of outliers in sample 2: {len(outliers_2)}")

    # -----------------------------------------------------------------------------------------------------------------

    # Set the bins to be from 0 to 1900 with step size 100
    bins = np.arange(0.2, 1.2, 0.075)

    # Plot histogram for sample_1 in blue color
    plt.hist(sample_1, bins, alpha=0.5, label='Sequential', color="firebrick")

    # Plot histogram for sample_2 in green color
    plt.hist(sample_2, bins, alpha=0.5, label='Parallel', color="deepskyblue")

    # Add legend
    plt.legend(loc='upper right')

    plt.title('Read CSV - Spended Time Distribution - 60.6mb')

    # Label the axes
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency')

    plt.savefig("read_csv_distribution_2.pdf", format='pdf', bbox_inches="tight")
    # Show the plot
    plt.show()

    # -----------------------------------------------------------------------------------------------------------------

    t_stat, p_value = ttest_ind(sample_1, sample_2)

    print('t-statistic:', t_stat)
    print('p-value:', p_value)

    # -----------------------------------------------------------------------------------------------------------------

    # let's assume sample_1 and sample_2 are your samples
    std_dev_1 = np.std(sample_1, ddof=1)
    std_dev_2 = np.std(sample_2, ddof=1)

    n1 = len(sample_1)
    n2 = len(sample_2)

    pooled_std_dev = np.sqrt(((n1 - 1) * std_dev_1**2 + (n2 - 1) * std_dev_2**2) / (n1 + n2 - 2))
    delta = 1
    sample_size = 84

    effect_size = delta / pooled_std_dev  # ratio of difference to standard deviation, delta is the minimum difference of interest
    alpha = 1 - 0.983  # significance level
    nobs = sample_size  # number of observations

    analysis = TTestIndPower()
    power = analysis.solve_power(effect_size = effect_size, nobs1 = nobs, alpha = alpha)

    print("Power of the test:", power, "Effect size:", effect_size)

    # =================================================================================================================

    print("CSV size: 209.4mb")

    sample_1 = data_pandas_read[2][:84]
    sample_2 = data_modin_read[2][:84]

    box_plot = plt.boxplot([sample_1, sample_2], vert=True, patch_artist=True, labels=['Sequential', 'Parallel'])

    colors = ["firebrick", "deepskyblue"]
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)

    plt.title('Box plot of Spended Time - 209.4mb')
    plt.ylabel('Read CSV - Spended Time')
    plt.xlabel('Computing Type')
    plt.savefig("read_csv_boxplot_3.pdf", format='pdf', bbox_inches="tight")
    plt.show()

    # -----------------------------------------------------------------------------------------------------------------

    # Calculate 50th percentile (median)
    median_1 = np.percentile(sample_1, 50)
    median_2 = np.percentile(sample_2, 50)

    print(f"50th percentile for sample 1: {median_1}")
    print(f"50th percentile for sample 2: {median_2}")

    # Identify outliers
    # Calculate the IQR for each sample
    iqr_1 = iqr(sample_1)
    iqr_2 = iqr(sample_2)

    # Define upper and lower bounds for outliers
    upper_bound_1 = np.percentile(sample_1, 75) + 1.5 * iqr_1
    lower_bound_1 = np.percentile(sample_1, 25) - 1.5 * iqr_1
    upper_bound_2 = np.percentile(sample_2, 75) + 1.5 * iqr_2
    lower_bound_2 = np.percentile(sample_2, 25) - 1.5 * iqr_2

    # Identify outliers
    outliers_1 = [x for x in sample_1 if x < lower_bound_1 or x > upper_bound_1]
    outliers_2 = [x for x in sample_2 if x < lower_bound_2 or x > upper_bound_2]

    print(f"Number of outliers in sample 1: {len(outliers_1)}")
    print(f"Number of outliers in sample 2: {len(outliers_2)}")

    # -----------------------------------------------------------------------------------------------------------------

    # Set the bins to be from 0 to 1900 with step size 100
    bins = np.arange(1, 7, 0.2)

    # Plot histogram for sample_1 in blue color
    plt.hist(sample_1, bins, alpha=0.5, label='Sequential', color="firebrick")

    # Plot histogram for sample_2 in green color
    plt.hist(sample_2, bins, alpha=0.5, label='Parallel', color="deepskyblue")

    # Add legend
    plt.legend(loc='upper right')

    plt.title('Read CSV - Spended Time Distribution - 209.4mb')

    # Label the axes
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency')

    plt.savefig("read_csv_distribution_3.pdf", format='pdf', bbox_inches="tight")
    # Show the plot
    plt.show()

    # -----------------------------------------------------------------------------------------------------------------

    t_stat, p_value = ttest_ind(sample_1, sample_2)

    print('t-statistic:', t_stat)
    print('p-value:', p_value)

    # -----------------------------------------------------------------------------------------------------------------

    # let's assume sample_1 and sample_2 are your samples
    std_dev_1 = np.std(sample_1, ddof=1)
    std_dev_2 = np.std(sample_2, ddof=1)

    n1 = len(sample_1)
    n2 = len(sample_2)

    pooled_std_dev = np.sqrt(((n1 - 1) * std_dev_1**2 + (n2 - 1) * std_dev_2**2) / (n1 + n2 - 2))
    delta = 1
    sample_size = 84

    effect_size = delta / pooled_std_dev  # ratio of difference to standard deviation, delta is the minimum difference of interest
    alpha = 1 - 0.983  # significance level
    nobs = sample_size  # number of observations

    analysis = TTestIndPower()
    power = analysis.solve_power(effect_size = effect_size, nobs1 = nobs, alpha = alpha)

    print("Power of the test:", power, "Effect size:", effect_size)

    # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    with open('pilot_study.npy', 'rb') as f:
        data = np.load(f)
        data = [array.tolist() for array in data]
        data = np.array(data)

    datasets = ["circles.csv",  "tripadvisor.csv", "taxi.csv"]

    data_pandas_read = [data[0, :, 2], data[1, :, 2], data[2, :, 2]]
    data_modin_read = [data[0, :, 3], data[1, :, 3], data[2, :, 3]]

    matplotlib.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots()
    fig.set_size_inches(11, 8)

    pandas_confidence_low = []
    pandas_confidence_high = []
    pandas_mean = []
    modin_confidence_low = []
    modin_confidence_high = []
    modin_mean = []

    for i in range(3):
        # Pandas
        confidence_lower, confidence_higher = st.t.interval(confidence=0.983, df=len(data_pandas_read[i])-1, loc=np.mean(data_pandas_read[i]), scale=st.sem(data_pandas_read[i]))
        print("Mean pandas:", np.mean(data_pandas_read[i]), "{} < p < {}".format(confidence_lower, confidence_higher), "Std:", np.std(data_pandas_read[i]))
        pandas_confidence_low.append(confidence_lower)
        pandas_confidence_high.append(confidence_higher)
        pandas_mean.append(np.mean(data_pandas_read[i]))
        
        # Modin
        confidence_lower, confidence_higher = st.t.interval(confidence=0.983, df=len(data_modin_read[i])-1, loc=np.mean(data_modin_read[i]), scale=st.sem(data_modin_read[i]))
        print("Mean modin:", np.mean(data_modin_read[i]), "{} < p < {}".format(confidence_lower, confidence_higher), "Std:", np.std(data_modin_read[i]))
        modin_confidence_low.append(confidence_lower)
        modin_confidence_high.append(confidence_higher)
        modin_mean.append(np.mean(data_modin_read[i]))
        
    ax.plot(["11.8kb", "60.6mb", "209.4mb"], pandas_mean, linestyle='-', linewidth=2, label="Sequential", c="firebrick")
    ax.fill_between(["11.8kb", "60.6mb", "209.4mb"], pandas_confidence_low, pandas_confidence_high, alpha=0.1, facecolor="firebrick")
    ax.plot(["11.8kb", "60.6mb", "209.4mb"], modin_mean, linestyle='-', linewidth=2, label="Parallel", c="deepskyblue")
    ax.fill_between(["11.8kb", "60.6mb", "209.4mb"], modin_confidence_low, modin_confidence_high, alpha=0.1, facecolor="deepskyblue")
    ax.legend(loc="lower left", mode="expand", ncol=3, prop={'size': 20})
    ax.set_xlabel('CSV size', fontsize=20)
    ax.set_ylabel('Time (s)', fontsize=20)
    ax.set_title('Concatenation Spended Time - Pilot study (n=10)', fontsize=24)
    ax.set_ylim([-0.4, 1.8])
    ax.grid()
    fig.set_dpi(100)
    plt.savefig("concatenation_pilot.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    # ----------------------------------------------------------------------------------------------------------------

    sample_1 = data_pandas_read
    sample_2 = data_modin_read

    n1 = len(sample_1)
    n2 = len(sample_2)

    print(np.mean(sample_1))
    print(np.mean(sample_2))

    # calculate standard deviation for each sample
    std_dev_1 = np.std(sample_1, ddof=1)  
    std_dev_2 = np.std(sample_2, ddof=1)

    # calculate pooled standard deviation
    pooled_std_dev = np.sqrt(((n1 - 1)*std_dev_1**2 + (n2 - 1)*std_dev_2**2) / (n1 + n2 - 2))
    print('Pooled standard deviation: ', pooled_std_dev)

    # ----------------------------------------------------------------------------------------------------------------

    delta = 1
    sd = pooled_std_dev

    # parameters for the analysis 
    effect_size = delta / sd  # ratio of difference to standard deviation, delta is the minimum difference of interest
    alpha = 1 - 0.983  # significance level
    power = 0.95   # desired power

    analysis = TTestIndPower()
    sample_size = analysis.solve_power(effect_size = effect_size, power = power, alpha = alpha)

    print(sample_size)

    # -----------------------------------------------------------------------------------------------------------------

    with open('data.npy', 'rb') as f:
        data = np.load(f)
        data = [array.tolist() for array in data]
        data = np.array(data)

    datasets = ["circles.csv",  "tripadvisor.csv", "taxi.csv"]

    data_pandas_read = [data[0, :, 2], data[1, :, 2], data[2, :, 2]]
    data_modin_read = [data[0, :, 3], data[1, :, 3], data[2, :, 3]]

    matplotlib.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots()
    fig.set_size_inches(11, 8)

    pandas_confidence_low = []
    pandas_confidence_high = []
    pandas_mean = []
    modin_confidence_low = []
    modin_confidence_high = []
    modin_mean = []

    for i in range(3):
        # Pandas
        confidence_lower, confidence_higher = st.t.interval(confidence=0.983, df=len(data_pandas_read[i][:12])-1, loc=np.mean(data_pandas_read[i][:12]), scale=st.sem(data_pandas_read[i][:12]))
        print("Mean pandas:", np.mean(data_pandas_read[i][:12]), "{} < p < {}".format(confidence_lower, confidence_higher), "Std:", np.std(data_pandas_read[i][:12]))
        pandas_confidence_low.append(confidence_lower)
        pandas_confidence_high.append(confidence_higher)
        pandas_mean.append(np.mean(data_pandas_read[i][:12]))
        
        # Modin
        confidence_lower, confidence_higher = st.t.interval(confidence=0.983, df=len(data_modin_read[i][:12])-1, loc=np.mean(data_modin_read[i][:12]), scale=st.sem(data_modin_read[i][:12]))
        print("Mean modin:", np.mean(data_modin_read[i][:12]), "{} < p < {}".format(confidence_lower, confidence_higher), "Std:", np.std(data_modin_read[i][:12]))
        modin_confidence_low.append(confidence_lower)
        modin_confidence_high.append(confidence_higher)
        modin_mean.append(np.mean(data_modin_read[i][:12]))
        
    ax.plot(["11.8kb", "60.6mb", "209.4mb"], pandas_mean, linestyle='-', linewidth=2, label="Sequential", c="firebrick")
    ax.fill_between(["11.8kb", "60.6mb", "209.4mb"], pandas_confidence_low, pandas_confidence_high, alpha=0.1, facecolor="firebrick")
    ax.plot(["11.8kb", "60.6mb", "209.4mb"], modin_mean, linestyle='-', linewidth=2, label="Parallel", c="deepskyblue")
    ax.fill_between(["11.8kb", "60.6mb", "209.4mb"], modin_confidence_low, modin_confidence_high, alpha=0.1, facecolor="deepskyblue")
    ax.legend(loc="lower left", mode="expand", ncol=3, prop={'size': 20})
    ax.set_xlabel('CSV size', fontsize=20)
    ax.set_ylabel('Time (s)', fontsize=20)
    ax.set_title('Concatenation - Spended Time', fontsize=24)
    ax.set_ylim([-0.4, 1.8])
    ax.grid()
    fig.set_dpi(100)
    plt.savefig("concatenation_csv.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    # ----------------------------------------------------------------------------------------------------------------

    print("CSV size: 209.4mb")

    sample_1 = data_pandas_read[2][:12]
    sample_2 = data_modin_read[2][:12]

    box_plot = plt.boxplot([sample_1, sample_2], vert=True, patch_artist=True, labels=['Sequential', 'Parallel'])

    colors = ["firebrick", "deepskyblue"]
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)

    plt.title('Box plot of Spended Time - 209.4mb')
    plt.ylabel('Concatenation - Spended Time')
    plt.xlabel('Computing Type')
    plt.savefig("concatenation_boxplot_3.pdf", format='pdf', bbox_inches="tight")
    plt.show()

    # -----------------------------------------------------------------------------------------------------------------

    # Calculate 50th percentile (median)
    median_1 = np.percentile(sample_1, 50)
    median_2 = np.percentile(sample_2, 50)

    print(f"50th percentile for sample 1: {median_1}")
    print(f"50th percentile for sample 2: {median_2}")

    # Identify outliers
    # Calculate the IQR for each sample
    iqr_1 = iqr(sample_1)
    iqr_2 = iqr(sample_2)

    # Define upper and lower bounds for outliers
    upper_bound_1 = np.percentile(sample_1, 75) + 1.5 * iqr_1
    lower_bound_1 = np.percentile(sample_1, 25) - 1.5 * iqr_1
    upper_bound_2 = np.percentile(sample_2, 75) + 1.5 * iqr_2
    lower_bound_2 = np.percentile(sample_2, 25) - 1.5 * iqr_2

    # Identify outliers
    outliers_1 = [x for x in sample_1 if x < lower_bound_1 or x > upper_bound_1]
    outliers_2 = [x for x in sample_2 if x < lower_bound_2 or x > upper_bound_2]

    print(f"Number of outliers in sample 1: {len(outliers_1)}")
    print(f"Number of outliers in sample 2: {len(outliers_2)}")

    # -----------------------------------------------------------------------------------------------------------------

    # Set the bins to be from 0 to 1900 with step size 100
    bins = np.arange(0, 2, 0.05)

    # Plot histogram for sample_1 in blue color
    plt.hist(sample_1, bins, alpha=0.5, label='Sequential', color="firebrick")

    # Plot histogram for sample_2 in green color
    plt.hist(sample_2, bins, alpha=0.5, label='Parallel', color="deepskyblue")

    # Add legend
    plt.legend(loc='upper right')

    plt.title('Concatenation - Spended Time Distribution - 209.4mb')

    # Label the axes
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency')

    plt.savefig("concatenation_distribution_3.pdf", format='pdf', bbox_inches="tight")
    # Show the plot
    plt.show()

    # -----------------------------------------------------------------------------------------------------------------

    t_stat, p_value = ttest_ind(sample_1, sample_2)

    print('t-statistic:', t_stat)
    print('p-value:', p_value)

    # -----------------------------------------------------------------------------------------------------------------

    # let's assume sample_1 and sample_2 are your samples
    std_dev_1 = np.std(sample_1, ddof=1)
    std_dev_2 = np.std(sample_2, ddof=1)

    n1 = len(sample_1)
    n2 = len(sample_2)

    pooled_std_dev = np.sqrt(((n1 - 1) * std_dev_1**2 + (n2 - 1) * std_dev_2**2) / (n1 + n2 - 2))
    delta = 1
    sample_size = 12

    effect_size = delta / pooled_std_dev  # ratio of difference to standard deviation, delta is the minimum difference of interest
    alpha = 1 - 0.983  # significance level
    nobs = sample_size  # number of observations

    analysis = TTestIndPower()
    power = analysis.solve_power(effect_size = effect_size, nobs1 = nobs, alpha = alpha)

    print("Power of the test:", power, "Effect size:", effect_size)

    # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    with open('pilot_study.npy', 'rb') as f:
        data = np.load(f)
        data = [array.tolist() for array in data]
        data = np.array(data)

    datasets = ["circles.csv",  "tripadvisor.csv", "taxi.csv"]

    data_pandas_read = [data[0, :, 4], data[1, :, 4], data[2, :, 4]]
    data_modin_read = [data[0, :, 5], data[1, :, 5], data[2, :, 5]]

    matplotlib.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots()
    fig.set_size_inches(11, 8)

    pandas_confidence_low = []
    pandas_confidence_high = []
    pandas_mean = []
    modin_confidence_low = []
    modin_confidence_high = []
    modin_mean = []

    for i in range(3):
        # Pandas
        confidence_lower, confidence_higher = st.t.interval(confidence=0.983, df=len(data_pandas_read[i])-1, loc=np.mean(data_pandas_read[i]), scale=st.sem(data_pandas_read[i]))
        print("Mean pandas:", np.mean(data_pandas_read[i]), "{} < p < {}".format(confidence_lower, confidence_higher), "Std:", np.std(data_pandas_read[i]))
        pandas_confidence_low.append(confidence_lower)
        pandas_confidence_high.append(confidence_higher)
        pandas_mean.append(np.mean(data_pandas_read[i]))
        
        # Modin
        confidence_lower, confidence_higher = st.t.interval(confidence=0.983, df=len(data_modin_read[i])-1, loc=np.mean(data_modin_read[i]), scale=st.sem(data_modin_read[i]))
        print("Mean modin:", np.mean(data_modin_read[i]), "{} < p < {}".format(confidence_lower, confidence_higher), "Std:", np.std(data_modin_read[i]))
        modin_confidence_low.append(confidence_lower)
        modin_confidence_high.append(confidence_higher)
        modin_mean.append(np.mean(data_modin_read[i]))
        
    ax.plot(["11.8kb", "60.6mb", "209.4mb"], pandas_mean, linestyle='-', linewidth=2, label="Sequential", c="firebrick")
    ax.fill_between(["11.8kb", "60.6mb", "209.4mb"], pandas_confidence_low, pandas_confidence_high, alpha=0.1, facecolor="firebrick")
    ax.plot(["11.8kb", "60.6mb", "209.4mb"], modin_mean, linestyle='-', linewidth=2, label="Parallel", c="deepskyblue")
    ax.fill_between(["11.8kb", "60.6mb", "209.4mb"], modin_confidence_low, modin_confidence_high, alpha=0.1, facecolor="deepskyblue")
    ax.legend(loc="lower left", mode="expand", ncol=3, prop={'size': 20})
    ax.set_xlabel('CSV size', fontsize=20)
    ax.set_ylabel('Time (s)', fontsize=20)
    ax.set_title('Dropna Spended Time - Pilot study (n=10)', fontsize=24)
    ax.set_ylim([-1.1, 9.5])
    ax.grid()
    fig.set_dpi(100)
    plt.savefig("dropna_pilot.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    # ----------------------------------------------------------------------------------------------------------------

    sample_1 = data_pandas_read
    sample_2 = data_modin_read

    n1 = len(sample_1)
    n2 = len(sample_2)

    print(np.mean(sample_1))
    print(np.mean(sample_2))

    # calculate standard deviation for each sample
    std_dev_1 = np.std(sample_1, ddof=1)  
    std_dev_2 = np.std(sample_2, ddof=1)

    # calculate pooled standard deviation
    pooled_std_dev = np.sqrt(((n1 - 1)*std_dev_1**2 + (n2 - 1)*std_dev_2**2) / (n1 + n2 - 2))
    print('Pooled standard deviation: ', pooled_std_dev)

    # ----------------------------------------------------------------------------------------------------------------

    delta = 3
    sd = pooled_std_dev

    # parameters for the analysis 
    effect_size = delta / sd  # ratio of difference to standard deviation, delta is the minimum difference of interest
    alpha = 1 - 0.983  # significance level
    power = 0.95   # desired power

    analysis = TTestIndPower()
    sample_size = analysis.solve_power(effect_size = effect_size, power = power, alpha = alpha)

    print(sample_size)

    # -----------------------------------------------------------------------------------------------------------------

    with open('data.npy', 'rb') as f:
        data = np.load(f)
        data = [array.tolist() for array in data]
        data = np.array(data)

    datasets = ["circles.csv",  "tripadvisor.csv", "taxi.csv"]

    data_pandas_read = [data[0, :, 4], data[1, :, 4], data[2, :, 4]]
    data_modin_read = [data[0, :, 5], data[1, :, 5], data[2, :, 5]]

    matplotlib.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots()
    fig.set_size_inches(11, 8)

    pandas_confidence_low = []
    pandas_confidence_high = []
    pandas_mean = []
    modin_confidence_low = []
    modin_confidence_high = []
    modin_mean = []

    for i in range(3):
        # Pandas
        confidence_lower, confidence_higher = st.t.interval(confidence=0.983, df=len(data_pandas_read[i][:32])-1, loc=np.mean(data_pandas_read[i][:32]), scale=st.sem(data_pandas_read[i][:32]))
        print("Mean pandas:", np.mean(data_pandas_read[i][:32]), "{} < p < {}".format(confidence_lower, confidence_higher), "Std:", np.std(data_pandas_read[i][:32]))
        pandas_confidence_low.append(confidence_lower)
        pandas_confidence_high.append(confidence_higher)
        pandas_mean.append(np.mean(data_pandas_read[i][:32]))
        
        # Modin
        confidence_lower, confidence_higher = st.t.interval(confidence=0.983, df=len(data_modin_read[i][:32])-1, loc=np.mean(data_modin_read[i][:32]), scale=st.sem(data_modin_read[i][:32]))
        print("Mean modin:", np.mean(data_modin_read[i][:32]), "{} < p < {}".format(confidence_lower, confidence_higher), "Std:", np.std(data_modin_read[i][:32]))
        modin_confidence_low.append(confidence_lower)
        modin_confidence_high.append(confidence_higher)
        modin_mean.append(np.mean(data_modin_read[i][:32]))
        
    ax.plot(["11.8kb", "60.6mb", "209.4mb"], pandas_mean, linestyle='-', linewidth=2, label="Sequential", c="firebrick")
    ax.fill_between(["11.8kb", "60.6mb", "209.4mb"], pandas_confidence_low, pandas_confidence_high, alpha=0.1, facecolor="firebrick")
    ax.plot(["11.8kb", "60.6mb", "209.4mb"], modin_mean, linestyle='-', linewidth=2, label="Parallel", c="deepskyblue")
    ax.fill_between(["11.8kb", "60.6mb", "209.4mb"], modin_confidence_low, modin_confidence_high, alpha=0.1, facecolor="deepskyblue")
    ax.legend(loc="lower left", mode="expand", ncol=3, prop={'size': 20})
    ax.set_xlabel('CSV size', fontsize=20)
    ax.set_ylabel('Time (s)', fontsize=20)
    ax.set_title('Dropna - Spended Time', fontsize=24)
    ax.set_ylim([-1.1, 9.5])
    ax.grid()
    fig.set_dpi(100)
    plt.savefig("dropna_csv.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    # -----------------------------------------------------------------------------------------------------------------

    print("CSV size: 11.8kb")

    sample_1 = data_pandas_read[0][:32]
    sample_2 = data_modin_read[0][:32]

    box_plot = plt.boxplot([sample_1, sample_2], vert=True, patch_artist=True, labels=['Sequential', 'Parallel'])

    colors = ["firebrick", "deepskyblue"]
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)

    plt.title('Box plot of Spended Time - 11.8kb')
    plt.ylabel('Dropna - Spended Time')
    plt.xlabel('Computing Type')
    plt.savefig("dropna_boxplot_1.pdf", format='pdf', bbox_inches="tight")
    plt.show()

    # -----------------------------------------------------------------------------------------------------------------

    # Calculate 50th percentile (median)
    median_1 = np.percentile(sample_1, 50)
    median_2 = np.percentile(sample_2, 50)

    print(f"50th percentile for sample 1: {median_1}")
    print(f"50th percentile for sample 2: {median_2}")

    # Identify outliers
    # Calculate the IQR for each sample
    iqr_1 = iqr(sample_1)
    iqr_2 = iqr(sample_2)

    # Define upper and lower bounds for outliers
    upper_bound_1 = np.percentile(sample_1, 75) + 1.5 * iqr_1
    lower_bound_1 = np.percentile(sample_1, 25) - 1.5 * iqr_1
    upper_bound_2 = np.percentile(sample_2, 75) + 1.5 * iqr_2
    lower_bound_2 = np.percentile(sample_2, 25) - 1.5 * iqr_2

    # Identify outliers
    outliers_1 = [x for x in sample_1 if x < lower_bound_1 or x > upper_bound_1]
    outliers_2 = [x for x in sample_2 if x < lower_bound_2 or x > upper_bound_2]

    print(f"Number of outliers in sample 1: {len(outliers_1)}")
    print(f"Number of outliers in sample 2: {len(outliers_2)}")

    # -----------------------------------------------------------------------------------------------------------------

    # Set the bins to be from 0 to 1900 with step size 100
    bins = np.arange(0, 0.2, 0.01)

    # Plot histogram for sample_1 in blue color
    plt.hist(sample_1, bins, alpha=0.5, label='Sequential', color="firebrick")

    # Plot histogram for sample_2 in green color
    plt.hist(sample_2, bins, alpha=0.5, label='Parallel', color="deepskyblue")

    # Add legend
    plt.legend(loc='upper right')

    plt.title('Dropna - Spended Time Distribution - 11.8kb')

    # Label the axes
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency')

    plt.savefig("dropna_distribution_1.pdf", format='pdf', bbox_inches="tight")
    # Show the plot
    plt.show()

    # -----------------------------------------------------------------------------------------------------------------

    t_stat, p_value = ttest_ind(sample_1, sample_2)

    print('t-statistic:', t_stat)
    print('p-value:', p_value)

    # -----------------------------------------------------------------------------------------------------------------

    # let's assume sample_1 and sample_2 are your samples
    std_dev_1 = np.std(sample_1, ddof=1)
    std_dev_2 = np.std(sample_2, ddof=1)

    n1 = len(sample_1)
    n2 = len(sample_2)

    pooled_std_dev = np.sqrt(((n1 - 1) * std_dev_1**2 + (n2 - 1) * std_dev_2**2) / (n1 + n2 - 2))
    delta = 3
    sample_size = 84

    effect_size = delta / pooled_std_dev  # ratio of difference to standard deviation, delta is the minimum difference of interest
    alpha = 1 - 0.983  # significance level
    nobs = sample_size  # number of observations

    analysis = TTestIndPower()
    power = analysis.solve_power(effect_size = effect_size, nobs1 = nobs, alpha = alpha)

    print("Power of the test:", power, "Effect size:", effect_size)

    # =================================================================================================================

    print("CSV size: 60.6mb")

    sample_1 = data_pandas_read[1][:32]
    sample_2 = data_modin_read[1][:32]

    box_plot = plt.boxplot([sample_1, sample_2], vert=True, patch_artist=True, labels=['Sequential', 'Parallel'])

    colors = ["firebrick", "deepskyblue"]
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)

    plt.title('Box plot of Spended Time - 60.6mb')
    plt.ylabel('Dropna - Spended Time')
    plt.xlabel('Computing Type')
    plt.savefig("dropna_boxplot_2.pdf", format='pdf', bbox_inches="tight")
    plt.show()

    # -----------------------------------------------------------------------------------------------------------------

    # Calculate 50th percentile (median)
    median_1 = np.percentile(sample_1, 50)
    median_2 = np.percentile(sample_2, 50)

    print(f"50th percentile for sample 1: {median_1}")
    print(f"50th percentile for sample 2: {median_2}")

    # Identify outliers
    # Calculate the IQR for each sample
    iqr_1 = iqr(sample_1)
    iqr_2 = iqr(sample_2)

    # Define upper and lower bounds for outliers
    upper_bound_1 = np.percentile(sample_1, 75) + 1.5 * iqr_1
    lower_bound_1 = np.percentile(sample_1, 25) - 1.5 * iqr_1
    upper_bound_2 = np.percentile(sample_2, 75) + 1.5 * iqr_2
    lower_bound_2 = np.percentile(sample_2, 25) - 1.5 * iqr_2

    # Identify outliers
    outliers_1 = [x for x in sample_1 if x < lower_bound_1 or x > upper_bound_1]
    outliers_2 = [x for x in sample_2 if x < lower_bound_2 or x > upper_bound_2]

    print(f"Number of outliers in sample 1: {len(outliers_1)}")
    print(f"Number of outliers in sample 2: {len(outliers_2)}")

    # -----------------------------------------------------------------------------------------------------------------

    # Set the bins to be from 0 to 1900 with step size 100
    bins = np.arange(0, 0.2, 0.01)

    # Plot histogram for sample_1 in blue color
    plt.hist(sample_1, bins, alpha=0.5, label='Sequential', color="firebrick")

    # Plot histogram for sample_2 in green color
    plt.hist(sample_2, bins, alpha=0.5, label='Parallel', color="deepskyblue")

    # Add legend
    plt.legend(loc='upper right')

    plt.title('Dropna - Spended Time Distribution - 60.6mb')

    # Label the axes
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency')

    plt.savefig("dropna_distribution_2.pdf", format='pdf', bbox_inches="tight")
    # Show the plot
    plt.show()

    # -----------------------------------------------------------------------------------------------------------------

    t_stat, p_value = ttest_ind(sample_1, sample_2)

    print('t-statistic:', t_stat)
    print('p-value:', p_value)

    # -----------------------------------------------------------------------------------------------------------------

    # let's assume sample_1 and sample_2 are your samples
    std_dev_1 = np.std(sample_1, ddof=1)
    std_dev_2 = np.std(sample_2, ddof=1)

    n1 = len(sample_1)
    n2 = len(sample_2)

    pooled_std_dev = np.sqrt(((n1 - 1) * std_dev_1**2 + (n2 - 1) * std_dev_2**2) / (n1 + n2 - 2))
    delta = 3
    sample_size = 84

    effect_size = delta / pooled_std_dev  # ratio of difference to standard deviation, delta is the minimum difference of interest
    alpha = 1 - 0.983  # significance level
    nobs = sample_size  # number of observations

    analysis = TTestIndPower()
    power = analysis.solve_power(effect_size = effect_size, nobs1 = nobs, alpha = alpha)

    print("Power of the test:", power, "Effect size:", effect_size)

    # =================================================================================================================

    print("CSV size: 209.4mb")

    sample_1 = data_pandas_read[2][:84]
    sample_2 = data_modin_read[2][:84]

    box_plot = plt.boxplot([sample_1, sample_2], vert=True, patch_artist=True, labels=['Sequential', 'Parallel'])

    colors = ["firebrick", "deepskyblue"]
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)

    plt.title('Box plot of Spended Time - 209.4mb')
    plt.ylabel('Dropna - Spended Time')
    plt.xlabel('Computing Type')
    plt.savefig("dropna_boxplot_3.pdf", format='pdf', bbox_inches="tight")
    plt.show()

    # -----------------------------------------------------------------------------------------------------------------

    # Calculate 50th percentile (median)
    median_1 = np.percentile(sample_1, 50)
    median_2 = np.percentile(sample_2, 50)

    print(f"50th percentile for sample 1: {median_1}")
    print(f"50th percentile for sample 2: {median_2}")

    # Identify outliers
    # Calculate the IQR for each sample
    iqr_1 = iqr(sample_1)
    iqr_2 = iqr(sample_2)

    # Define upper and lower bounds for outliers
    upper_bound_1 = np.percentile(sample_1, 75) + 1.5 * iqr_1
    lower_bound_1 = np.percentile(sample_1, 25) - 1.5 * iqr_1
    upper_bound_2 = np.percentile(sample_2, 75) + 1.5 * iqr_2
    lower_bound_2 = np.percentile(sample_2, 25) - 1.5 * iqr_2

    # Identify outliers
    outliers_1 = [x for x in sample_1 if x < lower_bound_1 or x > upper_bound_1]
    outliers_2 = [x for x in sample_2 if x < lower_bound_2 or x > upper_bound_2]

    print(f"Number of outliers in sample 1: {len(outliers_1)}")
    print(f"Number of outliers in sample 2: {len(outliers_2)}")

    # -----------------------------------------------------------------------------------------------------------------

    # Set the bins to be from 0 to 1900 with step size 100
    bins = np.arange(0, 10, 0.2)

    # Plot histogram for sample_1 in blue color
    plt.hist(sample_1, bins, alpha=0.5, label='Sequential', color="firebrick")

    # Plot histogram for sample_2 in green color
    plt.hist(sample_2, bins, alpha=0.5, label='Parallel', color="deepskyblue")

    # Add legend
    plt.legend(loc='upper right')

    plt.title('Dropna - Spended Time Distribution - 209.4mb')

    # Label the axes
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency')

    plt.savefig("dropna_distribution_3.pdf", format='pdf', bbox_inches="tight")
    # Show the plot
    plt.show()

    # -----------------------------------------------------------------------------------------------------------------

    t_stat, p_value = ttest_ind(sample_1, sample_2)

    print('t-statistic:', t_stat)
    print('p-value:', p_value)

    # -----------------------------------------------------------------------------------------------------------------

    # let's assume sample_1 and sample_2 are your samples
    std_dev_1 = np.std(sample_1, ddof=1)
    std_dev_2 = np.std(sample_2, ddof=1)

    n1 = len(sample_1)
    n2 = len(sample_2)

    print("agaragam:", sample_1, sample_2)

    pooled_std_dev = np.sqrt(((n1 - 1) * std_dev_1**2 + (n2 - 1) * std_dev_2**2) / (n1 + n2 - 2))
    delta = 3
    sample_size = 84

    effect_size = delta / pooled_std_dev  # ratio of difference to standard deviation, delta is the minimum difference of interest
    alpha = 1 - 0.983  # significance level
    nobs = sample_size  # number of observations

    analysis = TTestIndPower()
    power = analysis.solve_power(effect_size = effect_size, nobs1 = nobs, alpha = alpha)

    print("Power of the test:", power, "Effect size:", effect_size)

if __name__ == '__main__':
    main()
