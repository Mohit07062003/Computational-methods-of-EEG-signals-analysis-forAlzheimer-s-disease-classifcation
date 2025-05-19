import os
import pandas as pd
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mne

# Import functions from local modules
from src.data_loader import load_patient_data_subset, CHANNELS, EXPECTED_SIGNAL_LENGTH
from src.features import extract_all_features_for_subject, FS, FREQUENCY_BANDS # Import FREQUENCY_BANDS from features
# from src.preprocessing import FREQUENCY_BANDS as PREPROC_FREQUENCY_BANDS # Remove redundant import

# Import necessary components for statistics and ML
from scipy.stats import mannwhitneyu
from statsmodels.sandbox.stats.multicomp import multipletests
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline


# --- Configuration ---
# **UPDATE THIS LINE:** Path to the directory containing your EEG_data and results folders
BASE_PROJECT_PATH = 'C:\\Users\\mohit\\Desktop\\EEG-Project'

# Relative paths within the project directory
EEG_DATA_DIR_RELATIVE = 'EEG_data'
OUTPUT_DIR_RELATIVE = 'results'

# Construct absolute paths
BASE_EEG_DATA_PATH = os.path.join(BASE_PROJECT_PATH, EEG_DATA_DIR_RELATIVE)
OUTPUT_DIR = os.path.join(BASE_PROJECT_PATH, OUTPUT_DIR_RELATIVE)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Groups and Conditions in the dataset structure
DATA_GROUPS = ["Healthy", "AD"]
DATA_CONDITIONS = ["Eyes_open", "Eyes_closed"]

# Configuration for Subset Analysis
SUBJECT_FRACTION = 1 # Process only this fraction of subjects per group/condition
MIN_SUBJECTS_PER_GROUP_CONDITION = 2 # Ensure at least this many subjects per group/condition if possible

# Features to be analyzed and plotted
MEASURE_TYPES = ['C', 'F', 'Q', 'E', 'A', 'I'] # Include all 6 measure types


# Paper's comparison groups mapping internal names to paper names
COMPARISONS_CONFIG = [
    {'condition': 'Eyes_open', 'group1': 'Healthy', 'group2': 'AD', 'comp_name': 'A_vs_C'},
    {'condition': 'Eyes_closed', 'group1': 'Healthy', 'group2': 'AD', 'comp_name': 'B_vs_D'},
]

# --- Step 1: Load Subset Data and Extract Features ---
print("Step 1: Loading Subset Data and Extracting Features...")

# Load a subset of the data
print(f"Loading a subset of EEG data from {BASE_EEG_DATA_PATH}...")
eeg_dataset_subset = {}
for group in DATA_GROUPS:
    eeg_dataset_subset[group] = {}
    for condition in DATA_CONDITIONS:
        group_condition_path = os.path.join(BASE_EEG_DATA_PATH, group, condition)
        patients_list = []

        if not os.path.isdir(group_condition_path):
            print(f"Warning: Data path not found: {group_condition_path}. Skipping.")
            continue

        try:
            patient_folders = sorted(os.listdir(group_condition_path))
            # Select a fraction of folders, ensuring a minimum number
            num_to_select = max(MIN_SUBJECTS_PER_GROUP_CONDITION, int(len(patient_folders) * SUBJECT_FRACTION))
            selected_folders = patient_folders[:num_to_select]
            print(f"  Selected {len(selected_folders)} out of {len(patient_folders)} subjects for {group}/{condition}")

        except Exception:
             print(f"Error listing patient folders in {group_condition_path}. Skipping.")
             selected_folders = []

        for patient_folder in selected_folders:
            full_path = os.path.join(group_condition_path, patient_folder)
            if os.path.isdir(full_path):
                # Use the load_patient_data_subset from src.data_loader
                patient_data = load_patient_data_subset(full_path, CHANNELS, EXPECTED_SIGNAL_LENGTH)
                if any(data is not None for data in patient_data.values()):
                    patients_list.append({
                       "name": patient_folder,
                       "data": patient_data
                    })

        eeg_dataset_subset[group][condition] = patients_list

print("\nData loading complete for subset.")

# Extract features for the loaded subset
print("\nStarting feature extraction for subset...")
all_subjects_features = []

for group, conditions_data in eeg_dataset_subset.items():
    for condition, patients_list in conditions_data.items():
        print(f"Processing {group}/{condition} ({len(patients_list)} subjects)...")
        for patient_info in patients_list:
            patient_info['Group'] = group
            patient_info['Condition'] = condition
            # Call the function from src.features
            subject_features = extract_all_features_for_subject(patient_info, CHANNELS, FS, EXPECTED_SIGNAL_LENGTH)
            if len(subject_features) > 3: # Check if any features were added beyond Subject/Group/Condition
                all_subjects_features.append(subject_features)


features_df = pd.DataFrame(all_subjects_features)

# Drop columns where all values are NaN (if any features failed for all subjects)
features_df.dropna(axis=1, how='all', inplace=True)

# Save the extracted features DataFrame
features_save_path = os.path.join(OUTPUT_DIR, 'eeg_features_subset.pkl')
features_df.to_pickle(features_save_path)

print(f"\nFeature extraction complete for subset. Features saved to {features_save_path}")
print(f"Shape of the features DataFrame: {features_df.shape}")
print("Sample features DataFrame head:")
print(features_df.head())


# --- Step 2: Perform Statistical Analysis ---
print("\nStep 2: Performing Statistical Analysis (Mann-Whitney U test and AUC)...")

statistical_results = []

# Identify feature columns (excluding Subject, Group, Condition)
feature_types_prefixes = [f'{m}_' for m in MEASURE_TYPES]
feature_cols = [col for col in features_df.columns if any(col.startswith(prefix) for prefix in feature_types_prefixes)]

if features_df.empty or not feature_cols:
    print("No features available for statistical analysis. Skipping.")
else:
    print(f"Analyzing {len(feature_cols)} feature columns.")
    for comp in COMPARISONS_CONFIG:
        condition = comp['condition']
        group1_name = comp['group1'] # Healthy
        group2_name = comp['group2'] # AD
        comp_name = comp['comp_name']

        print(f"  Comparison: {comp_name} ({group1_name} vs {group2_name} under {condition})...")

        # Filter data for the current condition and groups
        df_filtered = features_df[features_df['Condition'] == condition].copy()
        df_group1 = df_filtered[df_filtered['Group'] == group1_name].copy()
        df_group2 = df_filtered[df_filtered['Group'] == group2_name].copy()

        # Need at least 2 subjects in each group for U test and AUC
        if len(df_group1) < 2 or len(df_group2) < 2:
            print(f"    Skipping comparison {comp_name}: Not enough subjects (Group1: {len(df_group1)}, Group2: {len(df_group2)}).")
            continue

        # Create combined group labels for AUC calculation (0 for group1, 1 for group2)
        group_labels = np.array([0] * len(df_group1) + [1] * len(df_group2))
        combined_df = pd.concat([df_group1, df_group2]).reset_index(drop=True)

        for col in feature_cols:
            # Extract measure, electrode, band from column name
            parts = col.split('_')
            measure = parts[0]
            if measure not in MEASURE_TYPES: continue # Skip if not one of our expected measures

            if measure == 'C':
                 if len(parts) < 3: continue
                 electrode = parts[1]
                 band_or_original = parts[2]
                 if band_or_original != 'Original': continue # C is only calculated for Original signal
            else:
                 if len(parts) < 2: continue
                 electrode = parts[1]
                 band_or_original = parts[2] if len(parts) > 2 else 'Original'
                 if band_or_original != 'Original' and band_or_original not in FREQUENCY_BANDS: continue # Skip unrecognized bands
                 if measure == 'E' and band_or_original == 'Original': continue # Skip Energy if Original


            feature_values = combined_df[col]
            valid_indices = feature_values.notna()

            values_valid = feature_values[valid_indices].values
            labels_valid = group_labels[valid_indices]

            values1_valid = values_valid[labels_valid == 0]
            values2_valid = values_valid[labels_valid == 1]

            if len(values1_valid) < 2 or len(values2_valid) < 2: # Need at least 2 points in each group for U-test/AUC
                 p_value = np.nan
                 auc_score = np.nan
            else:
                try:
                    stat, p_value = mannwhitneyu(values1_valid, values2_valid, alternative='two-sided')
                except ValueError:
                     p_value = np.nan
                except Exception:
                     p_value = np.nan

                if len(np.unique(labels_valid)) < 2:
                     auc_score = np.nan
                else:
                     try:
                        # Calculate AUC - Ensure positive class (AD) is consistently mapped to 1
                        # If group2_name is AD, then labels_valid=1 corresponds to AD. This is correct.
                        auc_score = roc_auc_score(labels_valid, values_valid)

                        # If a feature value is lower for AD than Healthy, AUC < 0.5.
                        # The paper doesn't explicitly flip AUC, but sometimes it's done to represent
                        # the strength of discrimination regardless of direction.
                        # Let's keep the raw AUC as per standard definition.
                        pass # Keep raw AUC

                     except ValueError:
                         auc_score = np.nan
                     except Exception:
                         auc_score = np.nan

            statistical_results.append({
                'Measure': measure,
                'Electrode': electrode,
                'Band': band_or_original,
                'Condition': condition,
                'Comparison': comp_name,
                'p_value': p_value,
                'AUC': auc_score
            })

    statistical_results_df = pd.DataFrame(statistical_results)
    statistical_results_df.dropna(subset=['p_value'], inplace=True) # Drop rows where test couldn't be performed
    statistical_results_df_sorted = statistical_results_df.sort_values(by='p_value', ascending=True).reset_index(drop=True)

    stats_save_path = os.path.join(OUTPUT_DIR, 'statistical_results_subset.csv')
    statistical_results_df_sorted.to_csv(stats_save_path, index=False)

    print("\nStatistical analysis complete.")
    print(f"Statistical results saved to {stats_save_path}")
    print("\nStatistical Results (sorted by p-value, top 10):")
    print(statistical_results_df_sorted.head(10))



# --- Step 3: Generate Visualizations (Scalp Plots and Boxplots) ---
print("\nStep 3: Generating Visualizations...")

# --- Scalp Plots (for per-channel features F, Q, E, A, Iota) ---
print("\nGenerating scalp plots...")

try:
    montage = mne.channels.make_standard_montage('standard_1020')
    ch_pos_raw = montage.get_positions()['ch_pos']
    plotting_channels_pos = {ch: (ch_pos_raw[ch][0], ch_pos_raw[ch][1])
                             for ch in CHANNELS if ch in ch_pos_raw}
    plotting_channels_list = list(plotting_channels_pos.keys())
    print(f"Loaded scalp layout for {len(plotting_channels_list)} channels.")
except Exception:
    print("Error loading MNE montage. Skipping scalp plots.")
    plotting_channels_pos = {}
    plotting_channels_list = []

def plot_scalp(values_dict: dict, channels_pos: dict, title: str, cmap='viridis_r', vmin=None, vmax=None, output_path=None):
    plot_channels_names = []
    plot_values = []
    plot_x_raw = []
    plot_y_raw = []

    for ch in channels_pos.keys():
        if ch in values_dict and values_dict[ch] is not None and not np.isnan(values_dict[ch]):
            plot_channels_names.append(ch)
            plot_values.append(values_dict[ch])
            plot_x_raw.append(channels_pos[ch][0])
            plot_y_raw.append(channels_pos[ch][1])

    if not plot_channels_names:
        return

    plot_x_raw = np.array(plot_x_raw)
    plot_y_raw = np.array(plot_y_raw)

    x_min, x_max = np.min(plot_x_raw), np.max(plot_x_raw)
    y_min, y_max = np.min(plot_y_raw), np.max(plot_y_raw)
    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
    x_scale, y_scale = (x_max - x_min) / 2, (y_max - y_min) / 2

    x_scale = max(x_scale, 1e-9)
    y_scale = max(y_scale, 1e-9)

    plot_x_norm = (plot_x_raw - x_center) / x_scale
    plot_y_norm = (plot_y_raw - y_center) / y_scale

    max_range = max(np.max(np.abs(plot_x_norm)), np.max(np.abs(plot_y_norm)))
    scaling_factor = 1.0 / max(max_range, 1e-9)

    plot_x_norm *= scaling_factor
    plot_y_norm *= scaling_factor

    distances_from_center = np.sqrt(plot_x_norm**2 + plot_y_norm**2)
    max_dist_norm = np.max(distances_from_center) if len(distances_from_center) > 0 else 0
    final_scaling = 0.8 / max(max_dist_norm, 1e-9)
    plot_x_norm *= final_scaling
    plot_y_norm *= final_scaling


    fig, ax = plt.subplots(figsize=(6, 6))

    head_radius = 1.0
    head_circle = plt.Circle((0, 0), head_radius, color='black', fill=False, linewidth=2, zorder=1)
    ax.add_patch(head_circle)

    nose_tip = (0, head_radius * 1.05)
    nose_base_l = (-head_radius * 0.05, head_radius * 0.95)
    nose_base_r = (head_radius * 0.05, head_radius * 0.95)
    ax.plot([nose_base_l[0], nose_tip[0], nose_base_r[0]], [nose_base_l[1], nose_tip[1], nose_base_r[1]], color='black', linewidth=2, zorder=1)

    ear_radius = head_radius * 0.08
    ear_y = 0
    ear_x = head_radius * 1.05
    left_ear = plt.Circle((-ear_x, ear_y), ear_radius, color='black', fill=False, linewidth=2, zorder=1)
    right_ear = plt.Circle((ear_x, ear_y), ear_radius, color='black', fill=False, linewidth=2, zorder=1)
    ax.add_patch(left_ear)
    ax.add_patch(right_ear)


    if 'p-value' in title.lower() and all(v is not None and not np.isnan(v) and v >= 0 for v in plot_values):
         log_values = -np.log10(np.array(plot_values) + 1e-100) # Add tiny epsilon for log(0)

         log_vmin = -np.log10(vmax if vmax is not None and vmax > 0 else 1.0)
         log_vmax = -np.log10(vmin if vmin is not None and vmin > 0 else np.min(plot_values) if plot_values else 1e-10)
         if log_vmin > log_vmax: log_vmin, log_vmax = log_vmax, log_vmin # Ensure vmin < vmax
         if log_vmin == log_vmax: log_vmax += 1 # Avoid zero range

         norm = plt.Normalize(vmin=log_vmin, vmax=log_vmax)
         cmap_inst = plt.get_cmap(cmap)
         colors = cmap_inst(norm(log_values))

         from matplotlib.ticker import FuncFormatter
         # Adjust tick levels to be sensible powers of 10 within the range
         tick_min_log = np.floor(log_vmin)
         tick_max_log = np.ceil(log_vmax)
         tick_locations_log = np.arange(tick_min_log, tick_max_log + 1)
         # Filter to be within the actual data range for better visualization
         tick_locations_log = tick_locations_log[(tick_locations_log >= log_vmin) & (tick_locations_log <= log_vmax)]

         formatter = FuncFormatter(lambda x, pos: f'{10**(-x):.0e}' if abs(x) >= 2 else f'{10**(-x):.2f}') # Format small p as sci, larger as decimal

    else:
         norm = plt.Normalize(vmin=vmin if vmin is not None else np.min(plot_values), vmax=vmax if vmax is not None else np.max(plot_values))
         cmap_inst = plt.get_cmap(cmap)
         colors = cmap_inst(norm(plot_values))
         from matplotlib.ticker import ScalarFormatter
         formatter = ScalarFormatter()

    ax.scatter(plot_x_norm, plot_y_norm, c=colors, s=200, edgecolors='black', zorder=5)

    for i, ch in enumerate(plot_channels_names):
        ax.text(plot_x_norm[i], plot_y_norm[i], ch, ha='center', va='center', fontsize=8)

    ax_range = head_radius * 1.2
    ax.set_xlim([-ax_range, ax_range])
    ax.set_ylim([-ax_range, ax_range])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    ax.axis('off')

    sm = plt.cm.ScalarMappable(cmap=cmap_inst, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

    if 'p-value' in title.lower() and all(v is not None and not np.isnan(v) and v >= 0 for v in plot_values):
         if tick_locations_log.size > 0:
            cbar.set_ticks(tick_locations_log)
            cbar.ax.set_yticklabels([formatter(loc, None) for loc in tick_locations_log])
         cbar.set_label('-log10(p-value)')
    else:
         if 'AUC' in title: cbar.set_label('AUC')
         elif 'Energy' in title: cbar.set_label('Relative Energy')
         else: cbar.set_label('Value')
         cbar.ax.yaxis.set_major_formatter(formatter)

    avg_value = np.nanmean(plot_values)
    if 'p-value' in title.lower():
         avg_text = f'Avg p-value: {avg_value:.2e}'
    elif 'AUC' in title:
         avg_text = f'Avg AUC: {avg_value:.3f}'
    else:
         avg_text = f'Avg: {avg_value:.3f}'
    fig.text(0.5, 0.03, avg_text, ha='center', fontsize=10)

    plt.tight_layout()

    if output_path:
        try:
            plt.savefig(output_path, dpi=300)
        except Exception:
            pass
        plt.close(fig)
    else:
        plt.show()


PER_CHANNEL_MEASURES = ['F', 'Q', 'E', 'A', 'I']
SCALP_PLOT_BANDS = ['Original'] + FREQUENCY_BANDS


if plotting_channels_pos and not statistical_results_df_sorted.empty:
    # Determine global p-value range for consistent coloring across all scalp plots
    all_p_values_for_scalp = statistical_results_df_sorted[
        statistical_results_df_sorted['Measure'].isin(PER_CHANNEL_MEASURES)
    ]['p_value'].dropna().values

    p_vmin_global = np.min(all_p_values_for_scalp) if len(all_p_values_for_scalp) > 0 else 1e-10
    p_vmax_global = 1.0
    p_vmin_global = max(p_vmin_global, 1e-30) # Set a floor to avoid extreme log values


    for measure in PER_CHANNEL_MEASURES:
        for band in SCALP_PLOT_BANDS:
            if measure == 'E' and band == 'Original': continue
            if measure != 'E' and band != 'Original' and band not in FREQUENCY_BANDS: continue


            for comp in COMPARISONS_CONFIG:
                condition = comp['condition']
                comp_name = comp['comp_name']

                plot_data_df = statistical_results_df_sorted[
                     (statistical_results_df_sorted['Measure'] == measure) &
                     (statistical_results_df_sorted['Band'] == band) &
                     (statistical_results_df_sorted['Condition'] == condition) &
                     (statistical_results_df_sorted['Comparison'] == comp_name)
                 ].copy()

                if 'Electrode' not in plot_data_df.columns:
                     continue

                p_values_dict = plot_data_df.set_index('Electrode')['p_value'].dropna().to_dict()

                if p_values_dict:
                    title = f"{measure} - {band} - {condition} ({comp_name}) - p-values"
                    save_filename = f"scalp_pvalue_{measure}_{band}_{condition}_{comp_name}".replace(' ', '_').replace('-', '_') + ".png"

                    plot_scalp(p_values_dict, plotting_channels_pos, title,
                               cmap='viridis_r', vmin=p_vmin_global, vmax=p_vmax_global,
                               output_path=os.path.join(OUTPUT_DIR, save_filename))


print("\nScalp plot generation complete.")

# --- Boxplots (for the "best" feature of each measure) ---
print("\nGenerating boxplots for the best feature of each measure...")

best_rows_by_measure = statistical_results_df_sorted[
    statistical_results_df_sorted['Measure'].isin(MEASURE_TYPES)
].groupby('Measure')['p_value'].idxmin()

best_features_for_boxplots = {} # {Measure: {Electrode: str, Band: str, Condition: str, Comparison: str, Feature_Col: str}}

if not statistical_results_df_sorted.empty:
    print("Identifying best feature for each measure based on lowest p-value...")
    for measure, index in best_rows_by_measure.items():
        best_row = statistical_results_df_sorted.loc[index]
        electrode = best_row['Electrode']
        band = best_row['Band']
        condition = best_row['Condition']
        comparison = best_row['Comparison']
        p_value = best_row['p_value']
        auc = best_row['AUC']

        if measure == 'C':
            feature_col = f'{measure}_{electrode}_{band}'
        else:
            feature_col = f'{measure}_{electrode}_{band}'

        if feature_col in features_df.columns:
             best_features_for_boxplots[measure] = {
                 'Electrode_Best': electrode,
                 'Band_Best': band,
                 'Condition_Best': condition,
                 'Comparison_Best': comparison,
                 'Feature_Col': feature_col,
                 'PValue': p_value,
                 'AUC': auc
             }
             print(f"  Best feature for {measure}: {feature_col} (p={p_value:.2e}, AUC={auc:.3f}, Comparison={comparison})")


def plot_boxplot_single_feature(df: pd.DataFrame, feature_col: str, group1_name: str, group2_name: str, group_labels_paper: List[str], title: str, ylabel: str, auc_value: float | None = None, output_path: str | None = None):
    df_plot = df[(df['Group'].isin([group1_name, group2_name]))].copy()

    if df_plot.empty or df_plot[feature_col].dropna().empty:
        return

    df_plot['Group'] = pd.Categorical(df_plot['Group'], categories=[group1_name, group2_name], ordered=True)

    fig, ax = plt.subplots(figsize=(4, 5))
    sns.boxplot(x='Group', y=feature_col, data=df_plot, order=[group1_name, group2_name], ax=ax)

    plt.xticks(ticks=[0, 1], labels=group_labels_paper)

    plt.title(title)
    plt.xlabel('')
    plt.ylabel(ylabel)

    if auc_value is not None and not np.isnan(auc_value):
         plt.text(0.5, 0.95, f'AUC = {auc_value:.3f}', horizontalalignment='center', verticalalignment='top', transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5))

    plt.tight_layout()

    if output_path:
        try:
            plt.savefig(output_path, dpi=300)
        except Exception:
            pass
        plt.close()
    else:
        plt.show()

if not features_df.empty and best_features_for_boxplots:
    print("\nGenerating boxplots for the best features...")
    for measure, feature_info in best_features_for_boxplots.items():
        feature_col = feature_info['Feature_Col']
        comparison = feature_info['Comparison_Best']
        auc_value = feature_info['AUC']
        p_value = feature_info['PValue']

        comp_details = None
        for comp in COMPARISONS_CONFIG:
             if comp['comp_name'] == comparison:
                  comp_details = comp
                  break
        if comp_details is None: continue

        group1_name = comp_details['group1']
        group2_name = comp_details['group2']
        if comp_details['condition'] == 'Eyes_open':
             group_labels_paper = ['A', 'C']
        elif comp_details['condition'] == 'Eyes_closed':
             group_labels_paper = ['B', 'D']
        else:
             group_labels_paper = [group1_name, group2_name]


        title = f"{measure} - {feature_info['Electrode_Best']} - {feature_info['Band_Best']} ({feature_info['Comparison_Best']})\np={p_value:.2e}"
        ylabel = measure

        df_plot_subset = features_df[
            (features_df['Condition'] == feature_info['Condition_Best']) &
            (features_df['Group'].isin([group1_name, group2_name]))
        ].copy()

        filename_parts = [measure, feature_info['Electrode_Best'], feature_info['Band_Best'], feature_info['Comparison_Best']]
        save_filename = "boxplot_" + "_".join(filename_parts).replace(' ', '_').replace('-', '_') + ".png"

        plot_boxplot_single_feature(df_plot_subset, feature_col, group1_name, group2_name, group_labels_paper, title, ylabel, auc_value, output_path=os.path.join(OUTPUT_DIR, save_filename))

print("\nBoxplot generation complete.")


# --- Step 4: Machine Learning Classification (Univariate SVM) ---
print("\nStep 4: Performing Machine Learning Classification (Univariate SVM)...")

classification_results = []

if not features_df.empty and best_features_for_boxplots:
    print("Performing univariate SVM classification for each measure's best feature (10-fold CV)...")

    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    comp_group_label_map = {
        'A_vs_C': {'condition': 'Eyes_open', 'group1_name': 'Healthy', 'group1_label': 0, 'group2_name': 'AD', 'group2_label': 1},
        'B_vs_D': {'condition': 'Eyes_closed', 'group1_name': 'Healthy', 'group1_label': 0, 'group2_name': 'AD', 'group2_label': 1},
    }

    for measure, best_feature_info in best_features_for_boxplots.items():
        feature_col = best_feature_info['Feature_Col']
        comparison = best_feature_info['Comparison_Best']
        best_condition = feature_info['Condition_Best']

        if comparison not in comp_group_label_map:
             continue

        comp_details = comp_group_label_map[comparison]
        group1_name = comp_details['group1_name']
        group2_name = comp_details['group2_name']
        group1_label = comp_details['group1_label']
        group2_label = comp_details['group2_label']

        df_classify = features_df[
            (features_df['Condition'] == best_condition) &
            (features_df['Group'].isin([group1_name, group2_name]))
        ].copy()

        X = df_classify[[feature_col]].values
        y = df_classify['Group'].apply(lambda x: group2_label if x == group2_name else group1_label).values

        valid_indices = np.where(~np.isnan(X).flatten())[0]
        X_valid = X[valid_indices]
        y_valid = y[valid_indices]

        unique_classes, counts = np.unique(y_valid, return_counts=True)
        if len(unique_classes) < 2:
             classification_results.append({
                 'Measure': measure, 'Electrode_Best': best_feature_info['Electrode_Best'],
                 'Band_Best': best_feature_info['Band_Best'], 'Condition_Best': best_condition,
                 'Comparison_Best': comparison,
                 'Acc_Mean (%)': np.nan, 'Acc_Std': np.nan, 'Sen_Mean (%)': np.nan, 'Sen_Std': np.nan,
                 'Spe_Mean (%)': np.nan, 'Spe_Std': np.nan, 'AUC_Mean': np.nan, 'AUC_Std': np.nan
             })
             continue

        min_samples_per_class = min(counts)
        if len(y_valid) < n_splits or min_samples_per_class < n_splits:
            classification_results.append({
                'Measure': measure, 'Electrode_Best': best_feature_info['Electrode_Best'],
                'Band_Best': best_feature_info['Band_Best'], 'Condition_Best': best_condition,
                'Comparison_Best': comparison,
                'Acc_Mean (%)': np.nan, 'Acc_Std': np.nan, 'Sen_Mean (%)': np.nan, 'Sen_Std': np.nan,
                'Spe_Mean (%)': np.nan, 'Spe_Std': np.nan, 'AUC_Mean': np.nan, 'AUC_Std': np.nan
            })
            continue

        model_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='rbf', probability=True, random_state=42))
        ])

        acc_scores = []
        sen_scores = []
        spe_scores = []
        auc_scores = []

        for fold_counter, (train_index, test_index) in enumerate(skf.split(X_valid, y_valid), 1):
            X_train, X_test = X_valid[train_index], X_valid[test_index]
            y_train, y_test = y_valid[train_index], y_valid[test_index]

            if len(np.unique(y_train)) < 2:
                 acc_scores.append(np.nan)
                 sen_scores.append(np.nan)
                 spe_scores.append(np.nan)
                 auc_scores.append(np.nan)
                 continue

            try:
                model_pipeline.fit(X_train, y_train)
                y_pred = model_pipeline.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                sensitivity = recall_score(y_test, y_pred, pos_label=group2_label, zero_division=0)
                specificity = recall_score(y_test, y_pred, pos_label=group1_label, zero_division=0)

                if len(np.unique(y_test)) > 1:
                     y_proba = model_pipeline.predict_proba(X_test)[:, 1]
                     auc = roc_auc_score(y_test, y_proba)
                else:
                     auc = np.nan

                acc_scores.append(acc)
                sen_scores.append(sensitivity)
                spe_scores.append(specificity)
                auc_scores.append(auc)

            except Exception:
                acc_scores.append(np.nan)
                sen_scores.append(np.nan)
                spe_scores.append(np.nan)
                auc_scores.append(np.nan)

        valid_acc_scores = [s for s in acc_scores if not np.isnan(s)]
        valid_sen_scores = [s for s in sen_scores if not np.isnan(s)]
        valid_spe_scores = [s for s in spe_scores if not np.isnan(s)]
        valid_auc_scores = [s for s in auc_scores if not np.isnan(s)]

        mean_acc = np.mean(valid_acc_scores) if valid_acc_scores else np.nan
        std_acc = np.std(valid_acc_scores) if len(valid_acc_scores) > 1 else 0.0

        mean_sen = np.mean(valid_sen_scores) if valid_sen_scores else np.nan
        std_sen = np.std(valid_sen_scores) if len(valid_sen_scores) > 1 else 0.0

        mean_spe = np.mean(valid_spe_scores) if valid_spe_scores else np.nan
        std_spe = np.std(valid_spe_scores) if len(valid_spe_scores) > 1 else 0.0

        mean_auc = np.mean(valid_auc_scores) if valid_auc_scores else np.nan
        std_auc = np.std(valid_auc_scores) if len(valid_auc_scores) > 1 else 0.0

        classification_results.append({
            'Measure': measure,
            'Electrode_Best': best_feature_info['Electrode_Best'],
            'Band_Best': best_feature_info['Band_Best'],
            'Condition_Best': best_condition,
            'Comparison_Best': comparison,
            'Acc_Mean (%)': mean_acc * 100 if not np.isnan(mean_acc) else np.nan,
            'Acc_Std': std_acc,
            'Sen_Mean (%)': mean_sen * 100 if not np.isnan(mean_sen) else np.nan,
            'Sen_Std': std_sen,
            'Spe_Mean (%)': mean_spe * 100 if not np.isnan(mean_spe) else np.nan,
            'Spe_Std': std_spe,
            'AUC_Mean': mean_auc,
            'AUC_Std': std_auc
        })

    classification_results_df = pd.DataFrame(classification_results)

    classification_save_path = os.path.join(OUTPUT_DIR, 'classification_results_subset.csv')
    classification_results_df.to_csv(classification_save_path, index=False)

    print("\nUnivariate SVM classification complete.")
    print(f"Classification results saved to {classification_save_path}")

    classification_results_display = classification_results_df[[
        'Measure', 'Electrode_Best', 'Band_Best', 'Condition_Best', 'Comparison_Best',
        'Acc_Mean (%)', 'Sen_Mean (%)', 'Spe_Mean (%)'
    ]].round(1) # Round percentages

    classification_results_display['AUC_Mean'] = classification_results_df['AUC_Mean'].round(3)
    print("\nClassification Results (Mean across folds, formatted like Table 2):")
    print(classification_results_display)

else:
    print("Skipping classification: No features or not enough data for analysis.")


print("\nSubset Analysis Complete.")
print(f"All results and plots saved in: {OUTPUT_DIR}")