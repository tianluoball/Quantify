import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.figure_factory as ff
import plotly.graph_objects as go

def calculate_confidence_interval(data, confidence=0.95):
    """Calculate confidence interval for a dataset"""
    n = len(data)
    # Ensure there is more than one data point to calculate sem
    if n <= 1:
        return (np.nan, np.nan)
    mean = np.mean(data)
    se = stats.sem(data)
    # Ensure degrees of freedom is greater than 0
    if n - 1 <= 0:
        return (np.nan, np.nan)
    ci = stats.t.interval(confidence, n-1, loc=mean, scale=se)
    return ci

def clean_numeric_data(series):
    """Clean data, handle special cases"""
    series = series.astype(str)
    series = series.str.replace(' .', '', regex=False)
    series = series.str.strip()
    return pd.to_numeric(series, errors='coerce')

def perform_wilcoxon_test(data1, data2):
    """Perform Wilcoxon signed-rank test"""
    # Wilcoxon test requires the same number of data points
    if len(data1) != len(data2):
        st.error("Wilcoxon signed-rank test requires the same number of samples for each group.")
        return np.nan, np.nan, np.nan
        
    statistic, p_value = stats.wilcoxon(data1, data2)
    n = len(data1)
    # Avoid division by zero if n=0
    if n == 0:
        return statistic, p_value, np.nan
    
    # The Z-score calculation is not straightforwardly derived from the statistic alone
    # and is often provided by more comprehensive statistical packages. 
    # This is a simplified approximation for effect size calculation.
    # We will use the rank-biserial correlation as effect size r = |(2S / n(n+1)) - 1|
    # where S is the Wilcoxon statistic.
    effect_size_r = abs((2 * statistic) / (n * (n + 1)) - 1)
    
    return statistic, p_value, effect_size_r


def calculate_group_comparison(group1_data, group2_data, confidence_decimal):
    """Calculate mean, median, quartiles comparison and percentage changes"""
    # Calculate means
    mean1 = group1_data.mean()
    mean2 = group2_data.mean()
    
    # Calculate medians and quartiles
    median1 = group1_data.median()
    median2 = group2_data.median()
    q1_1 = group1_data.quantile(0.25)
    q1_2 = group2_data.quantile(0.25)
    q3_1 = group1_data.quantile(0.75)
    q3_2 = group2_data.quantile(0.75)
    
    # Calculate absolute differences
    mean_abs_diff = mean2 - mean1
    median_abs_diff = median2 - median1
    q1_abs_diff = q1_2 - q1_1
    q3_abs_diff = q3_2 - q3_1
    
    # Calculate percentage changes
    def calc_pct_change(old_val, new_val):
        if old_val != 0:
            return ((new_val - old_val) / abs(old_val)) * 100
        return np.inf if new_val > 0 else -np.inf if new_val < 0 else 0
    
    mean_pct_change = calc_pct_change(mean1, mean2)
    median_pct_change = calc_pct_change(median1, median2)
    q1_pct_change = calc_pct_change(q1_1, q1_2)
    q3_pct_change = calc_pct_change(q3_1, q3_2)
    
    # Calculate confidence intervals
    ci1 = calculate_confidence_interval(group1_data.dropna(), confidence=confidence_decimal)
    ci2 = calculate_confidence_interval(group2_data.dropna(), confidence=confidence_decimal)
    
    return {
        'mean1': mean1,
        'mean2': mean2,
        'median1': median1,
        'median2': median2,
        'q1_1': q1_1,
        'q1_2': q1_2,
        'q3_1': q3_1,
        'q3_2': q3_2,
        'mean_abs_diff': mean_abs_diff,
        'mean_pct_change': mean_pct_change,
        'median_abs_diff': median_abs_diff,
        'median_pct_change': median_pct_change,
        'q1_abs_diff': q1_abs_diff,
        'q1_pct_change': q1_pct_change,
        'q3_abs_diff': q3_abs_diff,
        'q3_pct_change': q3_pct_change,
        'ci1': ci1,
        'ci2': ci2
    }

def main():
    st.title('Statistical Analysis Tool')
    
    uploaded_file = st.file_uploader("Upload Excel/CSV file", type=['xlsx', 'xls', 'csv'])
    
    confidence_level = st.number_input(
        "Enter Confidence Level (0-100)",
        min_value=1,
        max_value=99,
        value=95,
        step=1,
        help="Enter a number between 1 and 99. For example, enter 95 for 95% confidence level."
    )
    
    confidence_decimal = confidence_level / 100
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.write("Data Preview:")
            st.dataframe(df.head())
            
            all_columns = df.columns.tolist()
            
            st.subheader('Select Columns to Compare')
            col1, col2 = st.columns(2)
            
            with col1:
                st.write('Group 1')
                group1_cols = st.multiselect('Select columns', all_columns, key='group1')
            
            with col2:
                st.write('Group 2')
                group2_cols = st.multiselect('Select columns', all_columns, key='group2')
            
            col1, col2 = st.columns(2)
            t_test_button = col1.button('Perform T-Test')
            wilcoxon_button = col2.button('Perform Wilcoxon Test')
            
            if (t_test_button or wilcoxon_button) and group1_cols and group2_cols:
                try:
                    group1_data = df[group1_cols].apply(clean_numeric_data).mean(axis=1)
                    group2_data = df[group2_cols].apply(clean_numeric_data).mean(axis=1)
                    
                    valid_data = pd.DataFrame({
                        'group1': group1_data,
                        'group2': group2_data
                    }).dropna()
                    
                    if len(valid_data) > 0:
                        # Calculate group comparisons
                        comparison_results = calculate_group_comparison(
                            valid_data['group1'],
                            valid_data['group2'],
                            confidence_decimal
                        )
                        
                        # Display mean comparison results
                        st.subheader('Mean Comparison')
                        mean_df = pd.DataFrame({
                            'Metric': [
                                'Group 1 Mean',
                                f'Group 1 {int(confidence_level)}% Confidence Interval (Lower)',
                                f'Group 1 {int(confidence_level)}% Confidence Interval (Upper)',
                                'Group 2 Mean',
                                f'Group 2 {int(confidence_level)}% Confidence Interval (Lower)',
                                f'Group 2 {int(confidence_level)}% Confidence Interval (Upper)',
                                'Mean Absolute Difference (Group 2 - Group 1)',
                                'Mean Percentage Change'
                            ],
                            'Value': [
                                f"{comparison_results['mean1']:.4f}",
                                f"{comparison_results['ci1'][0]:.4f}",
                                f"{comparison_results['ci1'][1]:.4f}",
                                f"{comparison_results['mean2']:.4f}",
                                f"{comparison_results['ci2'][0]:.4f}",
                                f"{comparison_results['ci2'][1]:.4f}",
                                f"{comparison_results['mean_abs_diff']:.4f}",
                                f"{comparison_results['mean_pct_change']:.2f}%"
                            ]
                        })
                        st.dataframe(mean_df)

                        # Display median comparison results
                        st.subheader('Median Comparison')
                        median_df = pd.DataFrame({
                            'Metric': [
                                'Group 1 Median',
                                'Group 2 Median',
                                'Median Absolute Difference',
                                'Median Percentage Change'
                            ],
                            'Value': [
                                f"{comparison_results['median1']:.4f}",
                                f"{comparison_results['median2']:.4f}",
                                f"{comparison_results['median_abs_diff']:.4f}",
                                f"{comparison_results['median_pct_change']:.2f}%"
                            ]
                        })
                        st.dataframe(median_df)

                        # Display quartile comparison results
                        st.subheader('Quartile Comparison')
                        quartile_df = pd.DataFrame({
                            'Metric': [
                                'Group 1 Q1 (25th percentile)',
                                'Group 2 Q1 (25th percentile)',
                                'Q1 Absolute Difference',
                                'Q1 Percentage Change',
                                'Group 1 Q3 (75th percentile)',
                                'Group 2 Q3 (75th percentile)',
                                'Q3 Absolute Difference',
                                'Q3 Percentage Change'
                            ],
                            'Value': [
                                f"{comparison_results['q1_1']:.4f}",
                                f"{comparison_results['q1_2']:.4f}",
                                f"{comparison_results['q1_abs_diff']:.4f}",
                                f"{comparison_results['q1_pct_change']:.2f}%",
                                f"{comparison_results['q3_1']:.4f}",
                                f"{comparison_results['q3_2']:.4f}",
                                f"{comparison_results['q3_abs_diff']:.4f}",
                                f"{comparison_results['q3_pct_change']:.2f}%"
                            ]
                        })
                        st.dataframe(quartile_df)
                        
                        if t_test_button:
                            # T-test analysis
                            t_stat, p_value = stats.ttest_ind(valid_data['group1'], valid_data['group2'])
                            dof = len(valid_data['group1']) + len(valid_data['group2']) - 2
                            # Calculate Cohen's d
                            mean_diff = valid_data['group1'].mean() - valid_data['group2'].mean()
                            pooled_std = np.sqrt(((len(valid_data['group1']) - 1) * valid_data['group1'].var() + (len(valid_data['group2']) - 1) * valid_data['group2'].var()) / dof)
                            cohens_d = mean_diff / pooled_std
                            
                            st.subheader('T-Test Results')
                            results_df = pd.DataFrame({
                                'Statistic': ['T-statistic', 'P-value', 'Degrees of Freedom', "Cohen's d"],
                                'Value': [
                                    f"{t_stat:.4f}",
                                    f"{p_value:.4f}",
                                    f"{dof}",
                                    f"{cohens_d:.4f}"
                                ]
                            })
                            st.dataframe(results_df)
                            
                        elif wilcoxon_button:
                            # Wilcoxon test analysis
                            w_stat, w_pvalue, effect_size = perform_wilcoxon_test(
                                valid_data['group1'], 
                                valid_data['group2']
                            )
                            
                            if not np.isnan(w_stat):
                                st.subheader('Wilcoxon Test Results')
                                results_df = pd.DataFrame({
                                    'Statistic': ['Wilcoxon statistic', 'P-value', 'Effect size (r)'],
                                    'Value': [
                                        f"{w_stat:.4f}",
                                        f"{w_pvalue:.4f}",
                                        f"{effect_size:.4f}"
                                    ]
                                })
                                st.dataframe(results_df)
                                
                                # Effect size interpretation
                                st.subheader('Effect Size Interpretation')
                                if effect_size < 0.1:
                                    effect_text = "Negligible effect"
                                elif effect_size < 0.3:
                                    effect_text = "Small effect"
                                elif effect_size < 0.5:
                                    effect_text = "Medium effect"
                                else:
                                    effect_text = "Large effect"
                                st.write(f"Effect size interpretation: {effect_text}")
                                st.write(f"Statistical Significance: {'Significant' if w_pvalue < 0.05 else 'Not significant'} at α = 0.05")
                        
                        # Common visualizations
                        st.subheader('Descriptive Statistics')
                        stats_df = pd.DataFrame({
                            'Group': ['Group 1', 'Group 2'],
                            'Mean': [valid_data['group1'].mean(), valid_data['group2'].mean()],
                            'Median': [valid_data['group1'].median(), valid_data['group2'].median()],
                            'Std Dev': [valid_data['group1'].std(), valid_data['group2'].std()],
                            'Min': [valid_data['group1'].min(), valid_data['group2'].min()],
                            'Max': [valid_data['group1'].max(), valid_data['group2'].max()],
                            'N': [valid_data['group1'].count(), valid_data['group2'].count()]
                        })
                        st.dataframe(stats_df)
                        
                        # Box plot
                        st.subheader('Box Plot Comparison')
                        fig_box = go.Figure()
                        fig_box.add_trace(go.Box(y=valid_data['group1'], name='Group 1'))
                        fig_box.add_trace(go.Box(y=valid_data['group2'], name='Group 2'))
                        st.plotly_chart(fig_box)
                        
                        # Distribution plot (Side-by-side Bar Chart)
                        st.subheader('Distribution Plot')

                        # Determine a shared, appropriate bin range for both datasets
                        combined_data = pd.concat([valid_data['group1'], valid_data['group2']])
                        min_val = combined_data.min()
                        max_val = combined_data.max()
                        
                        # Use Freedman-Diaconis rule to determine optimal bin width, leading to bin number
                        iqr = combined_data.quantile(0.75) - combined_data.quantile(0.25)
                        n_combined = len(combined_data)
                        if iqr > 0 and n_combined > 0:
                            bin_width = (2 * iqr) / (n_combined ** (1/3))
                            num_bins = int((max_val - min_val) / bin_width)
                        else:
                            num_bins = 20 # Fallback to a default number of bins

                        # Calculate histograms for both groups using the same bins
                        hist1, bins = np.histogram(valid_data['group1'], bins=num_bins, range=(min_val, max_val))
                        hist2, _ = np.histogram(valid_data['group2'], bins=num_bins, range=(min_val, max_val))
                        
                        # Calculate the center of each bin for plotting
                        bin_centers = (bins[:-1] + bins[1:]) / 2

                        fig_dist = go.Figure()

                        # Add Bar trace for Group 1
                        fig_dist.add_trace(go.Bar(
                            x=bin_centers,
                            y=hist1,
                            name='Group 1'
                        ))

                        # Add Bar trace for Group 2
                        fig_dist.add_trace(go.Bar(
                            x=bin_centers,
                            y=hist2,
                            name='Group 2'
                        ))

                        # Update layout to group the bars side-by-side
                        fig_dist.update_layout(
                            barmode='group', # This is the key to making bars appear side-by-side
                            xaxis_title="Score",
                            yaxis_title="Frequency/Count",
                            showlegend=True,
                            bargap=0.15, # Optional: Adjusts the gap between bars of different groups
                            bargroupgap=0.1 # Optional: Adjusts the gap between bars within a group
                        )
                        st.plotly_chart(fig_dist)
                        
                    else:
                        st.error("No valid numerical data in selected columns after cleaning and removing missing values. Please check data format.")
                        
                except Exception as e:
                    st.error(f"Error performing analysis: {str(e)}")
                    st.write("Tip: Please ensure selected columns contain numerical data and are properly formatted.")
        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

if __name__ == '__main__':
    main()
