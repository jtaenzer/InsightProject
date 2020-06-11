# DB settings
database_name = "FutureFitAI_database"
collection_name = "talent_profiles"

# Where to dump binaries, leave as empty string if you don't want to save them
binary_path = "D:/FutureFit/classifying_tfidf_canada/"

min_skill_depth = 5000
min_title_freq = 250
min_skill_length = 10
train_test_frac = 0.05

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
                  #"realtor",  # pulled out because of high counts -- should be clustered later
                  ]