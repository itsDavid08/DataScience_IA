# %% Anomaly detection for classification
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from pycm import ConfusionMatrix

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

class AnomalyDetectorClassifier:
    """
    Parameters:
        data (DataFrame): The input dataset containing features and labels.
        contamination (float, optional): The proportion of outliers in the dataset. Default is 0.1.

    Attributes:
        data (DataFrame): The input dataset containing features and labels.
        contamination (float): The proportion of outliers in the dataset.
        detector (IsolationForest): The Isolation Forest model used for anomaly detection.
        x (DataFrame): Features extracted from the dataset.
        y (Series): Labels extracted from the dataset.
        X_train (DataFrame): Training features.
        X_test (DataFrame): Testing features.
        y_train (Series): Training labels.
        y_test (Series): Testing labels.
        y_pred (array-like): Predicted labels.

    Methods:
        __init__(data, contamination=0.1):
            Initializes the AnomalyDetectorClassifier object with input data and contamination parameter.
            It preprocesses the data, splits it into training and testing sets, and initializes the Isolation Forest model.

        _preprocess_data():
            Preprocesses the input data by extracting features (X) and labels (y).

        _produce_dataset():
            Splits the preprocessed data into training and testing sets.

        train():
            Trains the Isolation Forest model using the training data.

        evaluate():
            Evaluates the trained model on the testing data.
            Computes the confusion matrix and classification report.
            Returns the predicted labels.

        additional_metrics():
            Computes additional evaluation metrics using the confusion matrix.
            Prints the additional metrics.

        plot_analysis():
            Visualizes the analysis results by plotting violin plots for feature distributions by class,
            and a confusion matrix heatmap.

    Isolation Forest is an unsupervised learning algorithm for anomaly detection. It isolates outliers
    in the dataset by randomly selecting a feature and then randomly selecting a split value between the maximum
    and minimum values of that feature. This process is repeated recursively until the outliers are isolated
    into small partitions. The number of splits required to isolate an outlier serves as a measure of its anomaly score.

    Outliers detected by the Isolation Forest algorithm are likely the fraudulent transactions, thus Isolation Forest is
    used as a binary classifier, changing the autput so that 1 represents inliers (change to 0) and -1 represents
    outliers (change to 1).
    """

    def __init__(self, data, contamination=0.1):
        """
        Inicitiozes the atribustes and calls the methods to produce the data
        :param data: dataframe with the data to beexamined
        :param contamination: parameter to be considered for the Isolation Forest
        """
        self.data = data
        # Perform sanity check
        if not self._sanity_check():
            raise ValueError("Input data failed sanity check.")

        self.contamination = contamination
        self.detector = IsolationForest(contamination=self.contamination)
        self.x, self.y = self._preprocess_data()
        self.X_train, self.X_test, self.y_train, self.y_test = self._porduce_dataset()
        self.y_pred = 0

    def _sanity_check(self):
        """
        Performs a sanity check on the input data to ensure it meets basic requirements.

        Returns:
            bool: True if the data passes the sanity check, False otherwise.
        """
        # Check if the data is not empty
        if self.data.empty:
            print("Error: Input data is empty.")
            return False

        # Check if all columns are numeric
        if not self.data.select_dtypes(include=[np.number]).equals(self.data):
            print("Error: Not all columns are numeric.")
            return False

        # Check if the 'Class' column exists
        if 'Class' not in self.data.columns:
            print("Error: 'Class' column not found in the dataset.")
            return False

        # Check if the 'Class' column contains only binary labels (0 and 1)
        unique_classes = self.data['Class'].unique()
        if len(unique_classes) != 2 or set(unique_classes) != {0, 1}:
            print("Error: 'Class' column must contain only binary labels (0 and 1).")
            return False

        # Check for missing values in the dataset
        if self.data.isnull().values.any():
            print("Error: Input data contains missing values.")
            return False

        return True

    def _preprocess_data(self):
        """
        Preprocesses the input data by extracting features (X) and labels (y).

        Returns:
            tuple: A tuple containing the features (X) and labels (y).
        """
        # Last column contains the labels (0 for normal, 1 for anomaly)
        return self.data.iloc[:, :-1], self.data.iloc[:, -1]

    def _porduce_dataset(self):
        """
        Splits the preprocessed data into training and testing sets.

        Returns:
            tuple: A tuple containing the training and testing features and labels.
        """
        return train_test_split(self.x, self.y, test_size=0.2)

    def train(self):
        """
        Trains the Isolation Forest model using the training data.
        """
        self.detector.fit(self.X_train)

    def evaluate(self):
        """
        Evaluates the trained model on the testing data.
        Computes the confusion matrix and classification report.

        Returns:
            array-like: Predicted labels.
        """
        self.y_pred = self.detector.predict(self.X_test)
        self.y_pred[self.y_pred == 1] = 0  # 1 represents inliers, change to 0
        self.y_pred[self.y_pred == -1] = 1  # -1 represents outliers, change to 1
        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test, self.y_pred))
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.y_pred))
        return self.y_pred

    def additional_metrics(self):
        """
        Computes additional evaluation metrics using the confusion matrix.
        Prints the additional metrics.
        """
        cm = ConfusionMatrix(actual_vector=list(self.y_test), predict_vector=list(self.y_pred))
        print("Additional Metrics:")
        print(cm)

    def plot_analysis(self):
        """
        Visualizes the analysis results by plotting violin plots for feature distributions by class,
        and a confusion matrix heatmap.
        """
        features = self.data.columns[:-1]  # Exclude the 'Class' column
        fig, axs = plt.subplots(nrows=1, ncols=len(features), figsize=(6 * len(features), 6))

        for i, feature in enumerate(features):
            sns.violinplot(x='Class', y=feature, data=self.data, ax=axs[i], inner='quartile', split=True,
                           palette='muted')
            axs[i].set_title(f'Distribution of {feature}')
            axs[i].set_xlabel('Class')
            axs[i].set_ylabel(feature)

        plt.tight_layout()
        plt.show()

        cm = confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()


# Load the dataset
data = pd.read_csv('creditcard.csv')

# Instantiate the anomaly detector
anomaly_detector_classifier = AnomalyDetectorClassifier(data, 0.2)

# Train and evaluate the anomaly detector
anomaly_detector_classifier.train()
anomaly_detector_classifier.evaluate()
anomaly_detector_classifier.additional_metrics()
anomaly_detector_classifier.plot_analysis()

# %% Market Basket Analysis

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from mlxtend.frequent_patterns import apriori, association_rules
import warnings

warnings.filterwarnings("ignore")


class MarketBasketAnalysis:
    def __init__(self, data):
        """
        Initializes a MarketBasketAnalysis object with transactional data.

        :param data: DataFrame containing transactional data, where each row represents a transaction
        and each column represents an item.
        """
        self.data = data

    def clean_data(self):
        """
        Cleans the transactional data by removing duplicate rows, rows with missing values,
        and outliers based on numerical columns using the Isolation Forest algorithm.
        """
        # Remove lines that are duplicated or that have missing values
        self.data.drop_duplicates(inplace=True)
        self.data.dropna(inplace=True)
        # Drop rows where any value in a numerical column is less than or equal to 0, or if it is an outlier
        clf = IsolationForest(contamination=0.05)  # Initialize Isolation Forest model with a contamination rate of 0.05
        for col in self.data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                self.data = self.data[self.data[col] > 0]
                outlier_label = clf.fit_predict(self.data[[col]])
                self.data = self.data[outlier_label == 1]  # Non-outliers are marked as 1

    def explore_data(self, columns_to_explore, plot_2d_kde_plot=False):
        """
        Explores the transactional data by providing summary statistics and visualizing
        the distribution of specified columns.

        :param columns_to_explore: List of column names to explore.
        :param plot_2d_kde_plot: Boolean indicating whether to plot a 2D kernel density estimation plot.
        """
        for column_to_explore in columns_to_explore:
            print("Summary Statistics of column ", column_to_explore)
            print(self.data[column_to_explore].describe())

            # Visualize the column distribution
            plt.figure(figsize=(10, 6))
            sns.kdeplot(data=self.data[column_to_explore], log_scale=False, fill=True, bw_adjust=0.5)
            plt.title(f"Distribution of {column_to_explore}")
            plt.xlabel(column_to_explore)
            plt.ylabel("Density")
            plt.grid()
            plt.show()

        if plot_2d_kde_plot:
            plt.figure(figsize=(10, 6))
            sns.kdeplot(data=self.data, x=columns_to_explore[0], y=columns_to_explore[1],
                        fill=True, cmap='viridis', levels=10)
            plt.title("Relationship between UnitPrice and Quantity")
            plt.xlabel(columns_to_explore[0])
            plt.ylabel(columns_to_explore[1])
            plt.grid()
            plt.show()

    def find_association_rules(self, min_support=0.1, min_threshold=0.5):
        """
        Finds association rules in the transactional data using the Apriori algorithm.

        :param min_support: Minimum support threshold for frequent itemsets.
        :param min_threshold: Minimum threshold for association rules.
        :return: Tuple containing frequent itemsets and association rules.
        """
        # Constructs a pivot table where each row represents an invoice, each column represents a product description,
        # and the values represent the quantity of each product in the respective invoice.
        # lambda x: 1 if any(x.notna()) else 0 is a lambda function that checks if any value in the group is not NaN.
        # If at least one value is not NaN, it returns 1, indicating that the product was present in the invoice.
        # Otherwise, it returns 0.
        table = pd.pivot_table(self.data, values='Quantity', index=['InvoiceNo'],
                               columns=['Description'], aggfunc=lambda x: 1 if any(x.notna()) else 0, fill_value=0)
        # Apply the Apriori algorithm to mine frequent itemsets and generates association rules based on the frequent
        # itemsets found.
        frequent_itemsets = apriori(table, min_support=min_support, use_colnames=True)

        # Generate association rules, producing a dataframe with:
        # 'anntecedents': This column contains the antecedent itemsets of the association rules. Each itemset is
        # represented as a frozenset containing the items involved in the rule.
        # 'consequents': This column contains the consequent itemsets of the association rules. Similar to
        # 'antecedents', each itemset is represented as a frozenset containing the items involved in the rule.
        # For example, in a rule like {milk} -> {bread}, "milk" is the antecedent and "bread" is the consequent. This
        # rule implies that if "milk" is purchased, there's a high likelihood that "bread" will also be purchased.
        # 'antecedent support': This column represents the support of the antecedent itemsets. Support is the proportion
        # of transactions that contain the antecedent itemset.
        # 'consequent support': This column represents the support of the consequent itemsets. Similar to 'antecedent
        # support', it is the proportion of transactions that contain the consequent itemset.
        # 'support': This column represents the support of the association rule. Support is the proportion of
        # transactions that contain both the antecedent and consequent itemsets.
        # 'confidence': This column represents the confidence of the association rule. Confidence is the proportion of
        # transactions containing the antecedent itemset that also contain the consequent itemset.
        # 'lift': This column represents the lift of the association rule. Lift is the ratio of the observed support to
        # the expected support if the antecedent and consequent were independent. It indicates how much more likely the
        # consequent is given the antecedent, compared to if they were independent.
        # 'leverage': This column represents the leverage of the association rule. Leverage measures the difference
        # between the observed support and the expected support if the antecedent and consequent were independent. It
        # indicates the difference in frequency of occurrence of the antecedent and consequent together compared to what
        # would be expected if they were independent.
        # 'conviction': This column represents the conviction of the association rule. Conviction measures the ratio of
        # the expected frequency that the antecedent occurs without the consequent (if they were independent) to the
        # observed frequency of incorrect predictions.
        # 'zhangs_metric': This column represents Zhang's metric, which is a measure of the degree of dependency between
        # the antecedent and consequent in the association rule. It is based on the difference between the observed
        # confidence and the expected confidence under independence.
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_threshold)
        return frequent_itemsets, rules

    def explain_association_rules(self, rules):
        """
        Explains association rules, including symmetric rules, and prints their details.

        Note: it will print the rules, even the symmetric ones, for example:
        Rule #1: ABC -> CDE
          - Support: 0.0316
          - Confidence: 0.4848
          - Lift: 6.2634
        Rule #2: CDE -> ABC
          - Support: 0.0316
          - Confidence: 0.4085
          - Lift: 6.2634
        Are symmetric rules, in practical terms, these symmetric rules provide redundant information. One of the rules
        (usually the one with higher confidence or support) can be considered, while the other can be disregarded.

        :param rules: DataFrame containing association rules.
        :return:
        """
        print("Association Rules:")
        for idx, row in rules.iterrows():
            print(f"Rule #{idx + 1}: {list(row.iloc[0])} -> {list(row.iloc[1])}")
            print(f"  - Support: {row.iloc[4]:.4f}")
            print(f"  - Confidence: {row.iloc[5]:.4f}")
            print(f"  - Lift: {row.iloc[6]:.4f}")
            print(f"  - Leverage: {row.iloc[7]:.4f}")
            print(f"  - Conviction: {row.iloc[8]:.4f}")
            print(f"  - zHangs: {row.iloc[9]:.4f}")
            print()

    def visualize_association_rules(self, rules):
        """
        Visualizes association rules on a scatter plot, with support on the x-axis, confidence on the y-axis,
        and lift encoded as marker size.

        :param rules: DataFrame containing association rules.

        X-axis (Support): Support refers to the proportion of transactions in the dataset that contain the items present
        in a rule. In this plot, the X-axis represents the support values for the association rules.
        Y-axis (Confidence): Confidence measures how often a rule is found to be true. It's the probability of finding
        the consequent (the "then" part of the rule) in a transaction given that the transaction contains the antecedent
        (the "if" part of the rule). The Y-axis represents the confidence values for the association rules.
        Markers (Lift): Indicates the strength of association between antecedent and consequent in a rule. It helps us
        understand how much more likely the consequent is to occur when the antecedent is present compared to when it's
        not.

        Interpretation:
        Rules that are higher on the plot (towards the top-right corner) have both high support and high confidence.
        These rules are generally more significant and reliable.
        Rules that are towards the bottom-right corner have high confidence but relatively lower support.
        Rules towards the top-left corner have high support but lower confidence.
        Rules towards the bottom-left corner have both low support and low confidence.
        Bigger markers indicates that the antecedent and consequent appear together more often than expected based on
        their individual frequencies, meaning they are positively correlated.
        """
        plt.figure(figsize=(12, 8))

        # Define a color gradient based on support
        support_colors = sns.color_palette("viridis", as_cmap=True)

        # Encode support into marker size and color
        marker_size = rules['lift'] * 200
        marker_color = rules['lift']

        # Scatter plot with encoded support information
        plt.scatter(x='support', y='confidence', data=rules,
                    s=marker_size, c=marker_color, cmap=support_colors, alpha=0.6)

        # Add color bar for Lift
        plt.colorbar(label='Lift')

        # Add title and labels
        plt.title("Association Rules: Support vs Confidence", fontsize=16, fontname='Times New Roman')
        plt.xlabel("Support", fontsize=20, fontname='Times New Roman')
        plt.ylabel("Confidence", fontsize=20, fontname='Times New Roman')
        plt.xticks(fontsize=20, fontname='Times New Roman')
        plt.yticks(fontsize=20, fontname='Times New Roman')
        plt.show()

    def identify_interesting_patterns(self, rules, threshold):
        """
        Identifies interesting patterns in association rules based on a specified lift threshold.

        :param rules: DataFrame containing association rules.
        :param threshold: Minimum threshold for lift to consider a rule as interesting.
        """
        print("High Lift Rules:")
        print(self.explain_association_rules(rules[rules['lift'] > threshold]))


# Example usage
if __name__ == "__main__":
    # Load data (Using Online Retail dataset)
    data = pd.read_csv('Online_Retail.csv', parse_dates=['InvoiceDate'])
    # Initialize MarketBasketAnalysis object and examine only the relevant columns
    mba = MarketBasketAnalysis(data[['InvoiceNo', 'StockCode', 'Quantity', 'UnitPrice', 'Description']])
    # Clean and explore the data
    mba.clean_data()
    mba.explore_data(['Quantity', 'UnitPrice'], plot_2d_kde_plot=True)

    # Find association rules
    frequent_itemsets, rules = mba.find_association_rules(min_support=0.02, min_threshold=1)
    # Explain association rules
    mba.explain_association_rules(rules)
    # Visualize association rules
    mba.visualize_association_rules(rules)
    # Identify interesting patterns
    mba.identify_interesting_patterns(rules, 10)

#%% Hypothesis Testing
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, f_oneway, ttest_rel, wilcoxon, kruskal, friedmanchisquare, probplot, shapiro
from sklearn.datasets import load_iris

import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.formula.api import ols

class HypothesisTester:
    """
    The t-test assumes that the data is normally distributed and that the variances are equal between groups (for
    unpaired t-test) or within groups (for paired t-test).
    The ANOVA test assumes that the data is normally distributed and that the variances are equal between groups.
    """
    def unpaired_t_test(self, group1, group2):
        """
        Perform unpaired t-test for two groups.

        Parameters:
        - group1: List or array-like object containing data for group 1.
        - group2: List or array-like object containing data for group 2.

        Returns:
        - t_statistic: The calculated t-statistic.
        - p_value: The p-value associated with the t-statistic.
        """
        t_statistic, p_value = ttest_ind(group1, group2)
        return t_statistic, p_value

    def unpaired_anova(self, *groups):
        """
        Perform unpaired ANOVA for more than two groups.

        Parameters:
        - *groups: Variable length argument containing data for each group. Each argument should be a list or array-like
        object.

        Returns:
        - f_statistic: The calculated F-statistic.
        - p_value: The p-value associated with the F-statistic.
        """
        f_statistic, p_value = f_oneway(*groups)
        return f_statistic, p_value

    def paired_t_test(self, group1, group2):
        """
        Perform paired t-test for two groups.

        Parameters:
        - group1: List or array-like object containing data for group 1.
        - group2: List or array-like object containing data for group 2.
                  Should have the same length as group1.

        Returns:
        - t_statistic: The calculated t-statistic.
        - p_value: The p-value associated with the t-statistic.
        """
        t_statistic, p_value = ttest_rel(group1, group2)
        return t_statistic, p_value

    def paired_anova(self, data):
        """
        Perform paired ANOVA (repeated measures ANOVA) for more than two groups.

        Parameters:
        - data: Pandas DataFrame containing the data with columns representing different conditions.

        Returns:
        - f_statistic: The calculated F-statistic.
        - p_value: The p-value associated with the F-statistic.
        """
        model = ols('value ~ C(condition)', data=data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        return anova_table['F'][0], anova_table['PR(>F)'][0]

    def wilcoxon_ranksum_test(self, group1, group2):
        """
        Perform Wilcoxon rank-sum test (Mann-Whitney U test) for two independent samples.

        Parameters:
        - group1: List or array-like object containing data for sample 1.
        - group2: List or array-like object containing data for sample 2.

        Returns:
        - statistic: The calculated test statistic.
        - p_value: The p-value associated with the test statistic.
        """
        statistic, p_value = sms.stattools.stats.mannwhitneyu(group1, group2)

        return statistic, p_value

    def wilcoxon_signedrank_test(self, group1, group2):
        """
        Perform Wilcoxon signed-rank test for paired samples.
        Defines the alternative hypothesis with ‘greater’ option, this the distribution underlying d is stochastically
        greater than a distribution symmetric about zero; d represent the difference between the paired samples:
        d = x - y if both x and y are provided, or d = x otherwise.

        Parameters:
        - group1: List or array-like object containing data for sample 1.
        - group2: List or array-like object containing data for sample 2.
                  Should have the same length as group1.

        Returns:
        - statistic: The calculated test statistic.
        - p_value: The p-value associated with the test statistic.
        """
        statistic, p_value = wilcoxon(group1, group2, alternative="greater")
        return statistic, p_value

    def kruskal_wallis_test(self, *groups):
        """
        Perform Kruskal-Wallis H test for independent samples.

        Parameters:
        - *groups: Variable length argument containing data for each group. Each argument should be a list or array-like
        object.

        Returns:
        - statistic: The calculated test statistic.
        - p_value: The p-value associated with the test statistic.
        """
        statistic, p_value = kruskal(*groups)
        return statistic, p_value

    def friedman_test(self, *groups):
        """
        Perform Friedman test for related samples.

        Parameters:
        - *groups: Variable length argument containing data for each group. Each argument should be a list or array-like
        object representing measurements of the same individuals under different conditions.

        Returns:
        - statistic: The calculated test statistic.
        - p_value: The p-value associated with the test statistic.
        """
        statistic, p_value = friedmanchisquare(*groups)
        return statistic, p_value

    def qq_plots(self, variable_names, *data_samples, distribution='norm'):
        """
        Generate Q-Q plots for multiple data samples.

        Parameters:
        - *variable_names: List with the names of the variables to be plotted
        - data_samples: Variable number of 1D array-like objects representing the data samples.
        - distribution: String indicating the theoretical distribution to compare against. Default is 'norm' for normal
        distribution.

        Returns:
        - None (displays the Q-Q plots)
        """
        num_samples = len(data_samples)
        num_rows = (num_samples + 1) // 2  # Calculate the number of rows for subplots
        num_cols = 2 if num_samples > 1 else 1  # Ensure at least 1 column for subplots

        # Generate Q-Q plots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6))
        axes = axes.flatten()  # Flatten axes if multiple subplots

        for i, data in enumerate(data_samples):
            ax = axes[i]
            probplot(data, dist=distribution, plot=ax)
            ax.set_title(f'Q-Q Plot ({distribution})')
            ax.set_xlabel('Theoretical Quantiles')
            ax.set_ylabel(variable_names[i])

        # Adjust layout and show plots
        plt.tight_layout()
        plt.show()

    def test_normality(self, variable_names, *data_samples):
        """
        Test the normality of multiple data samples using Shapiro-Wilk test.

        Parameters:
        - variable_names: List with the names of the variables to be tested.
        - data_samples: Variable number of 1D array-like objects representing the data samples.

        Returns:
        - results: Dictionary containing the test results for each data sample.
                   The keys are the variable names and the values are a tuple (test_statistic, p_value) for
                   Shapiro-Wilk test.
        """
        results = {}
        for name, data in zip(variable_names, data_samples):
            results[name] = shapiro(data)
        for variable_name, shapiro_result in results.items():
            print(f'{variable_name}:')
            print(f'Shapiro-Wilk test - Test statistic: {shapiro_result.statistic}, p-value: {shapiro_result.pvalue}')
        return results

# Load the Iris dataset
iris = load_iris()
sepal_lengths = iris.data[:, 0]  # Sepal length data
sepal_widths = iris.data[:, 1]  # Sepal width data

# Sepal lengths for each species
setosa_lengths = sepal_lengths[iris.target == 0]
versicolor_lengths = sepal_lengths[iris.target == 1]
virginica_lengths = sepal_lengths[iris.target == 2]

# Sepal widths for each species
setosa_widths = sepal_widths[iris.target == 0]
versicolor_widths = sepal_widths[iris.target == 1]
virginica_widths = sepal_widths[iris.target == 2]

# Petal lengths for each species
petal_lengths = iris.data[:, 2]  # Petal length data
setosa_petals = petal_lengths[iris.target == 0]

# Initialize the HypothesisTester class with the data
tester = HypothesisTester()

# Perform normality analysis, first by visual checking using a Q-Q plot and then by normality test
tester.qq_plots(['setosa_lengths', 'versicolor_lengths', 'virginica_lengths', 'setosa_widths',
                 'versicolor_widths', 'virginica_widths'], setosa_lengths, versicolor_lengths,
                virginica_lengths, setosa_widths, versicolor_widths, virginica_widths)
tester.test_normality(['setosa_lengths', 'versicolor_lengths', 'virginica_lengths', 'setosa_widths',
                 'versicolor_widths', 'virginica_widths'], setosa_lengths, versicolor_lengths,
                virginica_lengths, setosa_widths, versicolor_widths, virginica_widths)

# Interpretation:
#
# Shapiro-Wilk Test:
# The Shapiro-Wilk test is a test of normality.
# The test statistic measures the discrepancy between the data and the normal distribution.
# The p-value indicates the probability of observing the data if the null hypothesis (data is normally distributed) is
# true.
# A higher p-value (closer to 1) suggests that the data is more likely to be normally distributed.
# A common significance level used to assess normality is 0.05. If the p-value is greater than 0.05, we fail to reject
# the null hypothesis and conclude that the data is approximately normally distributed.
# For setosa_lengths, versicolor_lengths, and virginica_lengths, the p-values are all greater than 0.05 (0.4595, 0.4647,
# 0.2583), indicating that we fail to reject the null hypothesis of normality. Therefore, we can conclude that these
# variables are approximately normally distributed.
# For setosa_widths, versicolor_widths, and virginica_widths, the p-values are all greater than 0.05 (0.2715, 0.3380,
# 0.1809), indicating that we fail to reject the null hypothesis of normality. Therefore, we can conclude that these
# variables are approximately normally distributed.

# Perform unpaired t-test between Setosa and Versicolor species
t_stat, p_val = tester.unpaired_t_test(setosa_lengths, versicolor_lengths)
print("\nUnpaired t-test between Setosa and Versicolor species:")
print("t-statistic:", t_stat)
print("p-value:", p_val)

# Perform unpaired ANOVA among all three species
f_stat, p_val_anova = tester.unpaired_anova(setosa_lengths, versicolor_lengths, virginica_lengths)
print("\nUnpaired ANOVA among all three species:")
print("F-statistic:", f_stat)
print("p-value:", p_val_anova)

# Perform paired t-test for Sepal length and Petal length within Setosa species
t_stat, p_val = tester.paired_t_test(setosa_lengths, setosa_petals)
print("\nPaired t-test for Sepal length and Petal length within Setosa species:")
print("t-statistic:", t_stat)
print("p-value:", p_val)

# Perform paired ANOVA for Sepal width within all three species
data = pd.DataFrame({
        'value': np.concatenate([setosa_widths, versicolor_widths, virginica_widths]),
        'condition': np.repeat(['setosa', 'versicolor', 'virginica'], len(setosa_widths))
    })
f_stat, p_val = tester.paired_anova(data)
print("\nPaired ANOVA for Sepal width within all three species:")
print("F-statistic:", f_stat)
print("p-value:", p_val)

# Perform Wilcoxon rank-sum test between Setosa and Versicolor species for sepal lengths
statistic, p_value = tester.wilcoxon_ranksum_test(setosa_lengths, versicolor_lengths)
print("\nWilcoxon rank-sum test between Setosa and Versicolor species for sepal lengths:")
print("Test statistic:", statistic)
print("p-value:", p_value)

# Perform Kruskal-Wallis test for Sepal width within all three species
statistic, p_value = tester.kruskal_wallis_test(setosa_widths, versicolor_widths, virginica_widths)
print("\nKruskal-Wallis test for Sepal width within all three species:")
print("Test statistic:", statistic)
print("p-value:", p_value)

# Perform Wilcoxon signed-rank test for Sepal length and Petal length within Setosa species
statistic, p_value = tester.wilcoxon_signedrank_test(setosa_lengths, setosa_petals)
print("\nWilcoxon signed-rank test for Sepal length and Petal length within Setosa species:")
print("Test statistic:", statistic)
print("p-value:", p_value)

# Perform Friedman test for Sepal length within all three species
statistic, p_value = tester.friedman_test(setosa_lengths, versicolor_lengths, virginica_lengths)
print("\nFriedman test for Sepal length within all three species:")
print("Test statistic:", statistic)
print("p-value:", p_value)

# Interpretation:
#
# Unpaired t-test between Setosa and Versicolor species:
# t-statistic: The calculated t-statistic is approximately -10.52 (indicates that, on average, the sepal lengths of
# Setosa species are lower than those of the Versicolor species). This value represents the difference in means between
# the sepal lengths of Setosa and Versicolor species.
# p-value: The p-value associated with the t-statistic is approximately 8.99e-18. This p-value is very small, indicating
# strong evidence against the null hypothesis.
# In this case, it suggests that the difference in sepal lengths between Setosa and Versicolor species is statistically
# significant.
#
# Unpaired ANOVA among all three species:
# F-statistic: The calculated F-statistic is approximately 119.26. This value represents the ratio of variability
# between groups to variability within groups in sepal lengths among all three species.
# p-value: The p-value associated with the F-statistic is approximately 1.67e-31. The t-test, this p-value is extremely
# small, indicating strong evidence against the null hypothesis.
# It suggests that there is a statistically significant difference in sepal lengths among at least one pair of species.
#
# Paired t-test for Sepal length and Petal length within Setosa species (used here just as a possible example as the
# samples as actually not paired):
# t-statistic: The calculated t-statistic is approximately 71.84. This value represents the difference in means between
# sepal lengths and petal lengths within the Setosa species relative to the variability in the data.
# p-value: The p-value associated with the t-statistic is approximately 2.54e-51. This p-value is extremely small,
# indicating strong evidence against the null hypothesis.
# It suggests that the difference between sepal lengths and petal lengths within the Setosa species is statistically
# significant.
#
# Paired ANOVA for Sepal width within all three species  (used here just as a possible example as the samples as
# actually not paired):
# F-statistic: The calculated F-statistic is approximately 49.16. This value represents the ratio of variability between
# groups to variability within groups in sepal widths among all three species.
# p-value: The p-value associated with the F-statistic is approximately 4.49e-17.This p-value is very small, indicating
# strong evidence against the null hypothesis.
# It suggests that there is a statistically significant difference in sepal widths among at least one pair of species.
#
# Wilcoxon rank-sum test between Setosa and Versicolor species for sepal lengths:
# Test statistic: The calculated test statistic is approximately 168.5. This value represents the difference in ranks
# between the two groups (Setosa and Versicolor) for sepal lengths.
# p-value: The p-value associated with the test statistic is approximately 8.35e-14. This small p-value indicates strong
# evidence against the null hypothesis.
# It suggests that there is a statistically significant difference in sepal lengths between Setosa and Versicolor
# species.
#
# Kruskal-Wallis test for Sepal width within all three species:
# Test statistic: The calculated test statistic is approximately 63.57. This value represents the sum of ranks across
# all groups for sepal widths.
# p-value: The p-value associated with the test statistic is approximately 1.57e-14. Similar to the previous result,
# this small p-value indicates strong evidence against the null hypothesis.
# It suggests that there is a statistically significant difference in sepal widths among the three species (Setosa,
# Versicolor, and Virginica).
#
# Wilcoxon signed-rank test for Sepal length and Petal length within Setosa species:
# Test statistic: The calculated test statistic is 1275.0. This value represents the sum of signed ranks of differences
# between the paired observations (sepal lengths and petal lengths) within the Setosa species.
# p-value: The p-value associated with the test statistic is approximately 8.88e-16. This small p-value indicates
# strong evidence against the null hypothesis.
# It suggests that there is a statistically significant difference between sepal lengths and petal lengths within the
# Setosa species.
#
# Friedman test for Sepal length within all three species:
# Test statistic: The calculated test statistic is approximately 73.79. This value represents the sum of ranks across
# all groups for sepal lengths.
# p-value: The p-value associated with the test statistic is approximately 9.50e-17. Once more, this small p-value
# indicates strong evidence against the null hypothesis.
# It suggests that there is a statistically significant difference in sepal lengths among the three species (Setosa,
# Versicolor, and Virginica).