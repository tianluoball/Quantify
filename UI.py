import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
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
    # Ensure the input is treated as a string for replacement operations
    series = series.astype(str)
    # The following line seems intended to remove trailing dots, but might be incorrect.
    # A more robust way would be to handle various non-numeric characters.
    # For now, keeping the original logic but noting it.
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

    # --- MODIFIED: Initialize session state for all persistent variables ---
    if 'analysis_ran' not in st.session_state:
        st.session_state.analysis_ran = False
    if 'plot_title' not in st.session_state:
        st.session_state.plot_title = "Distribution Comparison"
    if 'x_axis_title' not in st.session_state:
        st.session_state.x_axis_title = "Value"
    if 'y_axis_title' not in st.session_state:
        st.session_state.y_axis_title = "Count"
    if 'group1_legend' not in st.session_state:
        st.session_state.group1_legend = "Group 1"
    if 'group2_legend' not in st.session_state:
        st.session_state.group2_legend = "Group 2"
    
    uploaded_file = st.file_uploader("Upload Excel/CSV file", type=['xlsx', 'xls', 'csv'], key="file_uploader")
    
    # When a new file is uploaded, reset the analysis state
    if 'last_uploaded_file' not in st.session_state:
        st.session_state.last_uploaded_file = None
    if uploaded_file is not None and uploaded_file.file_id != st.session_state.last_uploaded_file:
        st.session_state.analysis_ran = False
        st.session_state.last_uploaded_file = uploaded_file.file_id

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

            if col1.button('Perform T-Test'):
                st.session_state.analysis_type = 't_test'
                st.session_state.analysis_ran = True
            
            if col2.button('Perform Wilcoxon Test'):
                st.session_state.analysis_type = 'wilcoxon'
                st.session_state.analysis_ran = True

            # --- MODIFIED: Main logic block now depends on session state ---
            if st.session_state.analysis_ran and group1_cols and group2_cols:
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
                        
                        if st.session_state.analysis_type == 't_test':
                            # T-test analysis
                            t_stat, p_value = stats.ttest_ind(valid_data['group1'], valid_data['group2'])
                            dof = len(valid_data['group1']) + len(valid_data['group2']) - 2
                            # Calculate Cohen's d
                            mean_diff = valid_data['group1'].mean() - valid_data['group2'].mean()
                            pooled_std = np.sqrt(((len(valid_data['group1']) - 1) * valid_data['group1'].var() + (len(valid_data['group2']) - 1) * valid_data['group2'].var()) / dof)
                            if pooled_std == 0:
                                cohens_d = np.nan # Avoid division by zero
                            else:
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
                        
                        elif st.session_state.analysis_type == 'wilcoxon':
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
                        fig_box.add_trace(go.Box(y=valid_data['group1'], name='Group 1', marker_color='#1f77b4'))
                        fig_box.add_trace(go.Box(y=valid_data['group2'], name='Group 2', marker_color='#ff7f0e'))
                        st.plotly_chart(fig_box)
                        
                        # Distribution Plot Section
                        st.subheader('Distribution Plot with Density Curve')
                        
                        # Use a form to prevent reruns on every character input
                        with st.form(key='plot_labels_form'):
                            st.write("Customize Plot Labels:")
                            col1, col2 = st.columns(2)
                            with col1:
                                plot_title_input = st.text_input("Plot Title", st.session_state.plot_title)
                                x_axis_title_input = st.text_input("X-Axis Title", st.session_state.x_axis_title)
                                y_axis_title_input = st.text_input("Y-Axis Title", st.session_state.y_axis_title)
                            with col2:
                                group1_legend_input = st.text_input("Group 1 Legend", st.session_state.group1_legend)
                                group2_legend_input = st.text_input("Group 2 Legend", st.session_state.group2_legend)
                            
                            submitted = st.form_submit_button("Update Plot Labels")
                            if submitted:
                                st.session_state.plot_title = plot_title_input
                                st.session_state.x_axis_title = x_axis_title_input
                                st.session_state.y_axis_title = y_axis_title_input
                                st.session_state.group1_legend = group1_legend_input
                                st.session_state.group2_legend = group2_legend_input
                                # No rerun needed, the script will flow down and redraw the plot with new labels


                        # Create figure with a single y-axis
                        fig_dist = go.Figure()

                        # Define colors
                        color1 = '#1f77b4'  # Blue
                        color2 = '#ff7f0e'  # Orange
                        
                        # Prepare data
                        group1_dist_data = valid_data['group1'].dropna()
                        group2_dist_data = valid_data['group2'].dropna()

                        # Determine binning strategy for histograms
                        if group1_dist_data.empty or group2_dist_data.empty:
                            st.warning("One or both groups have no data to plot.")
                            return
                        
                        combined_min = min(group1_dist_data.min(), group2_dist_data.min())
                        combined_max = max(group1_dist_data.max(), group2_dist_data.max())

                        # --- Calculate Histogram Data ---
                        bins = int(combined_max - combined_min) + 1
                        hist1_counts, hist1_bins = np.histogram(group1_dist_data, bins=bins, range=(combined_min, combined_max))
                        hist2_counts, hist2_bins = np.histogram(group2_dist_data, bins=bins, range=(combined_min, combined_max))
                        
                        # Get overall max count for y-axis range
                        max_count = max(hist1_counts.max(), hist2_counts.max()) if len(hist1_counts)>0 and len(hist2_counts)>0 else 0


                        # Add histogram traces (bars)
                        fig_dist.add_trace(
                            go.Histogram(
                                x=group1_dist_data, 
                                name=st.session_state.group1_legend, 
                                marker=dict(color=color1, line=dict(width=0)), # Remove bar outlines
                                xbins=dict(start=combined_min, end=combined_max, size=1)
                            )
                        )
                        fig_dist.add_trace(
                            go.Histogram(
                                x=group2_dist_data, 
                                name=st.session_state.group2_legend, 
                                marker=dict(color=color2, line=dict(width=0)), # Remove bar outlines
                                xbins=dict(start=combined_min, end=combined_max, size=1)
                            )
                        )

                        # --- Calculate and add Scaled KDE traces (lines) ---
                        x_range = np.linspace(combined_min, combined_max, 500)
                        max_density = 0
                        y1_kde, y2_kde = None, None

                        # Calculate KDE for Group 1 to find max density
                        try:
                            if len(group1_dist_data) > 1:
                                kde1 = stats.gaussian_kde(group1_dist_data)
                                y1_kde = kde1(x_range)
                                max_density = max(max_density, y1_kde.max())
                        except Exception: pass

                        # Calculate KDE for Group 2 to find max density
                        try:
                            if len(group2_dist_data) > 1:
                                kde2 = stats.gaussian_kde(group2_dist_data)
                                y2_kde = kde2(x_range)
                                max_density = max(max_density, y2_kde.max())
                        except Exception: pass
                        
                        # Calculate scaling factor
                        scaling_factor = (max_count / max_density) if max_density > 0 else 1

                        # Add scaled KDE traces
                        if y1_kde is not None:
                            fig_dist.add_trace(
                                go.Scatter(x=x_range, y=y1_kde * scaling_factor, mode='lines', line=dict(color=color1), showlegend=False)
                            )
                        if y2_kde is not None:
                             fig_dist.add_trace(
                                go.Scatter(x=x_range, y=y2_kde * scaling_factor, mode='lines', line=dict(color=color2), showlegend=False)
                            )

                        # Update layout with improved fonts and a single Y-axis
                        fig_dist.update_layout(
                            title_text=f'<b>{st.session_state.plot_title}</b>',
                            title_font_size=24,
                            barmode='group',
                            xaxis_title=f'<b>{st.session_state.x_axis_title}</b>',
                            xaxis_title_font_size=20,
                            yaxis_title=f'<b>{st.session_state.y_axis_title}</b>',
                            yaxis_title_font_size=20,
                            legend_title_text='',
                            font=dict(family="Arial Black, sans-serif"),
                            # --- MODIFIED: Tighter x-axis range ---
                            xaxis_range=[combined_min - 0.5, combined_max + 0.5]
                        )
                        
                        # Set y-axes titles and fonts with improved fonts and aligned zero
                        fig_dist.update_yaxes(
                            tickfont=dict(color='black', size=18, family='Arial Black'),
                            showgrid=True, gridwidth=1, gridcolor='LightGray',
                            range=[0, max_count * 1.15]
                        )
                        fig_dist.update_xaxes(
                            tickfont=dict(color='black', size=18, family='Arial Black'),
                            showgrid=False
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
