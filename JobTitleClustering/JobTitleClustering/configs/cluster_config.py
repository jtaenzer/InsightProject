# DB settings
database_name = "FutureFitAI_database"
collection_name = "talent_profiles_CAN"

# Where to dump binaries, leave as empty string if you don't want to save them
binary_path = ""

# How deep into the skill list do we want to go (sorted by value counts)
# This will establish the width of matrix input to the clustering
min_skill_depth = 5000
# How deep in the title list we want to to go (sorted by value counts)
# Dropping infrequent titles as a form of cleaning
min_title_freq = 5
# Minimum number of skills a profile has to contain to included in the data
min_skill_length = 10
# Subsample depth -- only used by cluster_subsample.py
subsample_depth = 200
# Number of clusters to stop the clustering at
n_cluster_stop = 1
# Distance measure to use, other options are "cosine", "l1", "cityblock" -- euclidean works best in small scale tests
affinity = "euclidean"
# Linkage strategy to use, other options are "single", "complete", "average" -- ward works best in small scale tests
linkage = "ward"
# List of "bad" job titles to drop from the data -- built by hand and should be subject to re-investigation
titles_to_drop = ["owner",
                  "founder",
                  "president",
                  "manager",
                  "partner",
                  "chief executive officer",
                  "director",
                  "consultant",
                  "retired",
                  "coordinator",
                  "supervisor",
                  "assistant",
                  "associate",
                  "leader",
                  "lead",
                  "buyer",
                  "employee",
                  "self employed"
                  ]
