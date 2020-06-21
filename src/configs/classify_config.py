# DB settings
database_name = "FutureFitAI_database"
collection_name = "talent_profiles_CAN"

# Where to dump binaries, leave as empty string if you don't want to save them
binary_path = "./binaries/"

min_skill_depth = 5000
min_title_freq = 100
min_skill_length = 5
# Fraction of data to use for training, the rest will be used for testing
train_test_frac = 0.8

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