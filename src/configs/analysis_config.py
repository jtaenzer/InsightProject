# Path to binaries to load
binary_path = "./binaries/"
# Path to save plots
plot_path = "./plots/"

# How many clusters do we want to find?
n_target_clusters = 200
# Do we want to ignore small clusters?
min_clus_size = 100
# Minimum purity -- to find good quality clusters we want them to be dominated by a particular title
min_purity = 10
# How deep into skills_series.value_counts() should we go when creating our core skills
core_skills_depth = 15