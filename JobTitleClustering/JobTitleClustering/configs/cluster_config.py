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
titles_to_drop = ["ceo",
                  "chief executive officer",
                  "president",
                  "pres",
                  "vice president",
                  "vp",
                  "senior vice president",
                  "executive vice president",
                  "assistant vice president",
                  "president & ceo",
                  "partner",  # Might actually refer to a law firm partner
                  "executive",
                  "executive director",
                  # Everything business owner related
                  "entrepreneur",
                  "owner",
                  "independent business owner",
                  "business owner",
                  "co-owner",
                  "co owner",
                  "founder",
                  "founder & ceo",
                  "co-founder",
                  "owner - operator",
                  "owner & ceo",
                  "president/ceo",
                  "chief/founder",
                  "brand manager / founder",
                  "owner/president",
                  "owner/operator",
                  "owner / operator",
                  "owner\manager",
                  "owner\operator",
                  "owner \ manager",
                  "owner / manager",
                  "owner / president",
                  "owner/manager",
                  "president / owner",
                  "owner\photographer",
                  "president/owner",
                  "business owner",
                  "small business owner",
                  # Management - might be worth clustering by themselves
                  "supervisor",
                  "assistant manager",
                  "store manager",
                  "project manager",
                  "branch manager",
                  "team lead",
                  "boss",
                  "director",
                  "manager",
                  "sales manager",
                  "operations manager",
                  "general manager",
                  "office manager",
                  # Assistants
                  "research assistant",
                  "executive assistant",
                  "assistant",
                  "administrative assistant",
                  "executive administrative assistant",
                  # Not really a job title
                  "retired",
                  "employee",
                  "disabled",
                  "mom",
                  "stay at home mom",
                  "housewife",
                  "homemaker",
                  "greater new york city area",  # lol
                  # Student
                  "summer student",
                  "phd candidate",
                  "graduate student",
                  "graduate research assistant",
                  # French
                  "chargée de projet",
                  "adjointe administrative",
                  "adjointe de direction",
                  "directeur",
                  "stagiaire",
                  "aucun",
                  "président",
                  "présidente",
                  "vice-présidente",
                  "chef d'entreprise",
                  "directeur général",
                  "directrice générale",
                  "superviseur",
                  "chargé de projet",
                  "chargé de projets",
                  "propriétaire",
                  "retraite",
                  "retraité",
                  "associé",
                  "gestionnaire de projets \ consultant en gestion",
                  "senior consultant \ consultante principale",
                  # Unsorted
                  "volunteer",
                  "intern",
                  "internship",
                  "associate",
                  "project coordinator",
                  "operator",
                  "administrator",
                  "office administrator",
                  "managing directory",
                  "buyer",
                  "research associate",
                  "member",
                  "consultant",
                  "senior consultant",
                  #"teacher",  # pulled out because of high counts -- should be clustered later
                  "realtor",  # pulled out because of high counts -- should be clustered later
                  "customer service representative", # pulled out because of high counts -- should be clustered later
                  "customer service rep", # pulled out because of high counts -- should be clustered later
                  "sales representative", # pulled out because of high counts -- should be clustered later
                  "sales rep", # pulled out because of high counts -- should be clustered later
                  ]
