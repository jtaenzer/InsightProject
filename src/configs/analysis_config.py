# Path to binaries to load
binary_path = "../binaries/120k_profiles/"
# Path to save plots
plot_path = "../plots/"

# More information will be printed out
verbose = True
# Create histograms and word clouds
plotting = True

# This should be the same as what was used for the clustering
min_skill_length = 10
# How many clusters do we want to find? **NOTE** this is a soft target and may never be reached
n_target_clusters = 200
# Do we want to ignore small clusters?
min_clus_size = 100
# Minimum purity -- to find "good" clusters we want them to be dominated by a particular title
# Purity = 100 * (most_frequent_title_count / cluster_size)
min_purity = 10
# How deep into skills_series.value_counts() should we go when creating our core skills
core_skills_depth = 15
# Training/testing fraction for classification
train_test_frac = 0.5