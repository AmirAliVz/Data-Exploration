


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import skew, mode
from scipy.stats import boxcox
import itertools
from scipy.stats import chi2_contingency
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

def plot_histogram_boxplot(data, filename, kde=True, display=True, figsize=(10, 6), fontsize=8, fontcolor='black'):

    """
    Plots a combined figure of a box plot and a histogram for a specified column of data in a Series.

    Parameters:
        data (pd.Series): The input column of data. Must be provided.
        filename (str): The name of the column to visualize. Must be provided.
        kde (bool, optional): Whether to use a kernel density estimator or not.
        display (bool): Whether or not to display the figure.
        ylog (bool, optional): If True, sets the histogram's y-axis to a logarithmic scale. Defaults to False.
        figsize (tuple, optional): The size of the figure shown. Defaults to (10, 6).
        fontsize (int, optional): Font size for annotations within the box plot. Defaults to 8.
        fontcolor (str, optional): Font color for annotations within the box plot. Defaults to 'black'.



    Returns:
        lower_bound (float): The lower bound used in the box plot for outlier detection.
        upper_bound (float): The upper bound used in the box plot for outlier detection.

    Note:
        The plot is saved as a .jpg image in the 'Figures' folder, named after the column with the specified suffix.
    """

    filename = filename.capitalize()
    # Calculate boxplot stats
    q1 = np.percentile(data, 25)
    median = np.median(data)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_whisker = max(data.min(), q1 - 1.5 * iqr)
    upper_whisker = min(data.max(), q3 + 1.5 * iqr)

    if not display:
        return lower_whisker, upper_whisker

    # Setup the figure
    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": (0.25, 0.75)}, sharex=True
    )

    cmap = plt.get_cmap('inferno')

    # --- Boxplot
    sns.boxplot(x=data, ax=ax_box, color='skyblue')

    # Annotate stats on boxplot
    box_stats = {
        'Min': lower_whisker,
        'Q1': q1,
        'Median': median,
        'Q3': q3,
        'Max': upper_whisker
    }

    # ------------------------
    for label, val in box_stats.items():
        ax_box.text(val, 0.02, f'{label}\n{val:.2f}',
                    ha='center', va='bottom', fontsize=fontsize,
                    color=fontcolor, rotation=45, weight='bold')

    ax_box.set(xlabel='')

    # --- Histogram
    hist_plot  = sns.histplot(data, bins='auto', kde=kde, color='steelblue', edgecolor='black', ax=ax_hist)

    # Annotate stats on boxplot
    hist_stats = {
        'Skew': skew(data),
        'STD': np.std(data),
        'Mode': mode(data, keepdims=True).mode[0],
        'Mean': np.mean(data)
    }

    # Example stats text
    stats_text = '\n'.join([
        f'Mean: {data.mean():.2f}',
        f'Mode: {data.mode().values[0]:.2f}',
        f'Std: {data.std():.2f}',
        f'Skew: {data.skew():.2f}'
    ])

    # Draw rectangle (manual position and size in Axes coordinates: 0–1 range)
    bbox_props = dict(boxstyle="round4,pad=1.2", fc="seashell", ec="black", alpha=0.5)
    ax_hist.text(
        0.85, 0.6, stats_text,
        transform=ax_hist.transAxes,
        verticalalignment='top', horizontalalignment='right',
        bbox=bbox_props, ha='center',
        va='bottom', fontsize=fontsize+2, color=fontcolor, weight='bold'
    )

    # Accessing the patches (bars) and bins from the plot
    patches = hist_plot.patches  # These are the bars of the histogram

    # To get the counts, you can use `patches` to calculate the heights (counts)
    n = [patch.get_height() for patch in patches]

    # Gradient fill on histogram bars
    for patch, count in zip(patches, n):
        color = cmap(0.3 + 0.7 * count / max(n))
        patch.set_facecolor(color)

    # Annotate count values on bars
    for patch, count in zip(patches, n):
        if count > 0:
            ax_hist.text(patch.get_x() + patch.get_width() / 2,
                         count,
                         f'{int(count)}',
                         ha='center', va='bottom', fontsize=8)

    # Titles and labels
    ax_hist.set_title(f'Distribution of {filename}', fontweight='bold')
    ax_hist.set_xlabel(filename, fontweight='bold')
    ax_hist.set_ylabel('Count', fontweight='bold')

    plt.tight_layout()

    # Check if 'Figures' folder exists, and if not, create it
    if not os.path.exists("Figures"):
        os.makedirs("Figures")

    plt.savefig('Figures/' + filename + '.jpg', dpi=300)
    if verbose:
        plt.show()
    else:
        plt.close()

    return lower_whisker, upper_whisker

def piechart(data, filename, verbose=True):
    """
    Generates a pie chart from the provided data column and saves the figure.

    Parameters:
    -----------
    data : pandas.Series or list-like
        The data column containing categorical values to be plotted in the pie chart.

    filename : str
        The name to use when saving the pie chart image file (without path).
    """

    filename = filename.capitalize()
    # Get frequency counts
    counts = data.value_counts()
    counts.index = counts.index.str.capitalize()
    # Determine the mode
    mode_sex = counts.idxmax()

    # Pie chart enhancements
    plt.figure(figsize=(7, 7))
    colors = ['#ff9999', '#66b3ff']  # Custom colors
    explode = [0.08 if sex == mode_sex else 0 for sex in counts.index]  # Explode the mode for emphasis
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=colors, startangle=90,
            explode=explode, shadow=True,
            textprops={'fontsize': 20, 'color': 'black', 'fontweight': 'bold'},
            wedgeprops={'edgecolor': 'black', 'linewidth': 2, 'alpha': 0.8}
            )

    plt.title('Distribution of ' + filename, fontsize=18, fontweight='bold')

    # Check if 'Figures' folder exists, and if not, create it
    if not os.path.exists("Figures"):
        os.makedirs("Figures")

    plt.savefig('Figures/' + filename + '.jpg', dpi=300)
    if verbose:
        plt.show()
    else:
        plt.close()

def barchart(data, filename, fontsize=14, verbose=True):
    """
    Generates and saves a bar chart based on the frequency of categorical data.

    Parameters:
    -----------
    data : pandas.Series or list-like
        The data column containing categorical values to be plotted in the bar chart.

    filename : str
        The name of the output image file to be saved in the 'Figures' directory (without path).

    fontsize : int, optional (default=14)
        Font size for axis labels and tick marks.

    verbose : bool, optional (default=True)
        If True, displays the plot after saving. If False, suppresses plot display.
    """

    filename = filename.capitalize()
    # Get frequency counts
    counts = data.value_counts()
    counts.index = counts.index.str.capitalize()

    # Set figure size
    plt.figure(figsize=(8, 6))

    # Define custom colors using Seaborn's palette
    colors = sns.color_palette("pastel")[:len(counts)]

    # Create bar chart with custom aesthetics
    bars = plt.bar(counts.index, counts, color=colors,
                   edgecolor='black', linewidth=1.5)

    # Add labels on top of bars
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, 1.03*bar.get_height(), f'{int(bar.get_height())}',
                 ha='center', fontsize=fontsize, fontweight='bold', color='black')

        prc = round(bar.get_height()/len(data)*100, 1)
        plt.text(bar.get_x() + bar.get_width()/2, 0.5*bar.get_height(), f'{float(prc)}' + '%',
                 ha='center', fontsize=fontsize, fontweight='bold', color='black')

    # Style the chart
    plt.title('Distribution of ' + filename, fontsize=fontsize+4, fontweight='bold', color='#333333')
    plt.xlabel(filename, fontsize=fontsize+2, fontweight='bold', color='#555555')
    plt.ylabel('Frequency Count', fontsize=fontsize+2, fontweight='bold', color='#555555')

    # Adjust grid aesthetics
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Remove top and right borders for a cleaner look
    plt.box(False)

    # Check if 'Figures' folder exists, and if not, create it
    if not os.path.exists("Figures"):
        os.makedirs("Figures")

    # Save & show the plot
    plt.savefig('Figures/' + filename + '.jpg', dpi=300)
    if verbose:
        plt.show()
    else:
        plt.close()

def scatter_pearson(x, y, c1, c2, verbose=True):
    """
        Creates a scatter plot of two continuous variables and computes the Pearson correlation coefficient.

        Parameters:
        -----------
        x : array-like or pandas.Series
            Values for the x-axis (independent variable).

        y : array-like or pandas.Series
            Values for the y-axis (dependent variable).

        c1 : str
            Label for the x-axis (e.g., variable name).

        c2 : str
            Label for the y-axis (e.g., variable name).

        verbose : bool, optional (default=True)
            If True, displays the scatter plot with the correlation.

        Returns:
        --------
        pearson_corr : float
            The computed Pearson correlation coefficient between x and y.
        """

    c1 = c1.capitalize()
    c2 = c2.capitalize()

    """Applies Box-Cox transformation if skewness is high."""
    if abs(skew(x)) >= 1:
        x, best_lambda = boxcox(x + 1)
    if abs(skew(y)) >= 1:
        y, best_lambda = boxcox(y + 1)

    # Compute correlation
    correlation_matrix = np.corrcoef(x, y)

    # Extract correlation coefficient
    pearson_corr = correlation_matrix[0, 1]
    if verbose:
        print(f'Pearson Correlation of {c1} & {c2}: {pearson_corr:.2f}')

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.regplot(x=x, y=y, scatter_kws={'alpha':0.5})

    # Draw rectangle (manual position and size in Axes coordinates: 0–1 range)
    bbox_props = dict(boxstyle="round4,pad=1.2", fc="seashell", ec="black", alpha=0.5)

    plt.text(
        0.9, 0.8, f'r: {pearson_corr:.2f}',
        transform=ax.transAxes,
        verticalalignment='top', horizontalalignment='right',
        bbox=bbox_props, ha='center',
        va='bottom', fontsize=14 + 2, color='black', weight='bold'
    )

    plt.title(c1+' vs '+c2, fontsize=16, fontweight='bold')
    plt.xlabel(c1, weight='bold')
    plt.ylabel(c2, weight='bold')

    # Check if 'Figures' folder exists, and if not, create it
    if not os.path.exists("Figures"):
        os.makedirs("Figures")

    plt.savefig('Figures/' + c1+' vs '+c2 + '.jpg', dpi=300)
    if verbose:
        plt.show()
    else:
        plt.close()

    return pearson_corr

def two_categorical(x, y, col1, col2, verbose=True):
    """
    Performs a Chi-Square test of independence between two categorical variables,
    and generates visualizations for interpretation.

    Parameters:
    -----------
    x : pandas.Series
        The first categorical variable.

    y : pandas.Series
        The second categorical variable.

    col1 : str
        Name or label for the first variable (used in plots and output titles).

    col2 : str
        Name or label for the second variable.

    verbose : bool, optional (default=True)
        If True, displays:
            - A heatmap of the contingency table
            - Side-by-side bar charts for distribution comparison
            - Test results summary

    Returns:
    --------
    chi2 : float
        The Chi-Square test statistic.

    p : float
        The p-value of the test.
    """


    col1 = col1.capitalize()
    col2 = col2.capitalize()

    # Assuming your dataset is in a pandas Series
    cross_tab = pd.crosstab(x, y)
    if verbose:
        print(cross_tab)


    # Perform Chi-squared test
    chi2, p, dof, expected = chi2_contingency(cross_tab)
    if verbose:
        print(f"Chi-squared value of {col1} & {col2}: {chi2}")
        print(f"P-value: {p}")
        print(50*'=')

    # Setup the figure
    fig, (ax_heat, ax_bar) = plt.subplots(
        2, 1, figsize=(10,8))

    # Barplot
    sns.countplot(x=x, hue=y, ax=ax_bar)
    ax_bar.set_xlabel(col1, fontsize=14, fontweight='bold')  # X-axis label
    ax_bar.set_ylabel(col2 + ' Frequency', fontsize=14, fontweight='bold')  # Y-axis label

    # Heatmap of the contingency table
    sns.heatmap(cross_tab, annot=True, cmap='Blues', fmt='d', ax=ax_heat)
    ax_heat.set_title('Heatmap & Grouped BarChart of '+col1+' & '+col2 +
                      '\n \n chi2: ' + str(format(chi2, ".2e")) + '    -    p-value: ' + str(format(p, ".2e")), fontsize=16, fontweight='bold')  # Title
    ax_heat.set_xlabel(col2, fontsize=14, fontweight='bold')  # X-axis label
    ax_heat.set_ylabel(col1, fontsize=14, fontweight='bold')  # Y-axis label

    # Check if 'Figures' folder exists, and if not, create it
    if not os.path.exists("Figures"):
        os.makedirs("Figures")

    plt.savefig('Figures/' + col1 + ' vs ' + col2 + '.jpg', dpi=300)
    if verbose:
        plt.show()
    else:
        plt.close()

    return chi2, p

def bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        return 'Healthy weight'
    elif bmi < 30:
        return 'Overweight'
    else:
        return 'Obesity'


# Change global font to 'Arial', size 14
plt.rcParams['font.family'] = 'Georgia'
plt.rcParams['font.size'] = 12

verbose = True
# Read the 'Employee Turnover Dataset' CSV file into a DataFrame
df = pd.read_excel("data/Health Insurance Dataset.xlsx")
# Clean the tail
df_clean = df[:-10].copy()

# Converting data types to their intended type
df_clean = df_clean.convert_dtypes()
if verbose:
    print("=" * 50)
    print(" Data Types ")
    print("=" * 50)
    print(df_clean.dtypes)

# Adding the additional info on obesity ranges
# Apply the function to create a new column
df_clean['bmi_category'] = df_clean['bmi'].apply(bmi_category)

# =======================================================================
# Part I: Univariate and Bivariate Statistical Analysis and Visualization
# =======================================================================
# 1. Univariate Statistical Analysis and Visualization
#  ---- Two Continuous Variables (e.g., age, charges)
if verbose:
    print("=" * 50)
    print(" 1. Univariate -- Two Continuous Variables ")
    print("=" * 50)

# Visualization of histogram
columnName = ['age', 'bmi', 'charges', 'score']
for col in columnName:
    plot_histogram_boxplot(df_clean[col], col, fontsize=12)

#  ---- Two Categorical Variables (e.g., sex, region)
if verbose:
    print("=" * 50)
    print(" 1. Univariate -- Two Categorical Variables ")
    print("=" * 50)

# Showing the Region distribution by barchart
columnName = 'bmi_category'
reg_data = df_clean[columnName]
barchart(reg_data, columnName, verbose=verbose)

# Showing Sex distribution by pie chart
columnName = 'sex'
sex_data = df_clean[columnName]
piechart(sex_data, columnName, verbose=verbose)

# =====================================================
# 2. Bivariant Statistical Analysis and Visualization
if verbose:
    print("=" * 50)
    print(" 2. Bivariant -- Two Continuous Variables ")
    print("=" * 50)

#  ---- Two Continuous Variables (e.g., score, charges)
columnName = ['age', 'bmi', 'charges', 'score']

# Generate all unique bivariate combinations
combinations = list(itertools.combinations(columnName, 2))

# Scatter Plot + Pearson Correlation ===============
for i, (col_a, col_b) in enumerate(combinations, 1):
    scatter_pearson(df_clean[col_a], df_clean[col_b], col_a, col_b, verbose=verbose)

#  ---- Two Categorical Variables (e.g., score, charges)
if verbose:
    print("=" * 50)
    print(" 2. Bivariant -- Two Categorical Variables ")
    print("=" * 50)

columnName = ['sex', 'smoker', 'region', 'Level', 'bmi_category']

# Generate all unique bivariate combinations
combinations = list(itertools.combinations(columnName, 2))

# Heatmap & Grouped BarChart =======================
for i, (col_a, col_b) in enumerate(combinations, 1):
    two_categorical(df_clean[col_a], df_clean[col_b], col_a, col_b, verbose=verbose)

# =======================================
# Part II: Parametric Statistical Testing
# =======================================
if verbose:
    print("=" * 50)
    print(" Part II: Parametric -- t-test (sex vs. BMI) ")
    print("=" * 50)

# Independent Two-Sample t-test ==================
# Separate age by smoker status
col1 = 'sex'
col2 = 'bmi'
bmi_male = df_clean[df_clean[col1] == 'male'][col2]
bmi_female = df_clean[df_clean[col1] == 'female'][col2]

# Perform independent two-sample t-test
t_stat, p_value = stats.ttest_ind(bmi_male, bmi_female)

# Print results
print("T-statistic:", t_stat)
print("P-value:", p_value)

if verbose:
    # Set figure size and style
    plt.figure(figsize=(14, 6))

    # === Violin Plot ===
    plt.subplot(1, 2, 1)
    sns.violinplot(x='sex', y='bmi', hue='sex',
                   data=df_clean, inner='quartile',
                   palette='Set2', dodge=False, legend=False)
    plt.title('BMI Distribution by Sex (Violin Plot)', fontsize=14, fontweight='bold')
    plt.xlabel('Sex', fontsize=14, fontweight='bold')
    plt.ylabel('BMI', fontsize=14, fontweight='bold')

    # === Histogram + KDE Overlay ===
    plt.subplot(1, 2, 2)
    sns.histplot(data=df_clean, x='bmi', hue='sex', kde=True,
                 element='step', stat='density', common_norm=False,palette='Set1')
    plt.title('BMI Distribution by Sex (Histogram + KDE)', fontsize=14, fontweight='bold')
    plt.xlabel('BMI', fontsize=14, fontweight='bold')
    plt.ylabel('Density', fontsize=14, fontweight='bold')

    # Display both plots
    plt.tight_layout()
    plt.savefig('Figures/' + 'Sex' + ' vs ' + 'BMI' + '.jpg', dpi=300)
    plt.show()



# =======================================
# Part III: Nonparametric Statistical Testing
# =======================================
if verbose:
    print("=" * 50)
    print(" Part III: Nonparametric -- Chi-square test ")
    print("=" * 50)

# Create a contingency table
contingency_table = pd.crosstab(df_clean['region'], df_clean['bmi_category'])

# Run the Chi-square test
chi2, p, dof, expected = chi2_contingency(contingency_table)

print(20*'=' + 'Chi2-Test region & bmi_category' + 20*'=')
print(f"Chi2 Statistic: {chi2:.4f}, p-value: {p:.4g}, Degrees of Freedom: {dof}")
