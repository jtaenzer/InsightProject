# Cleaning/encoding settings
create_encoding = False
encoding_path = "D:/FutureFit/"
skill_count_depth = 10000   # Tunes the number of skills to keep, if depth = X we will keep the X most frequent skills
encoding_name = "classes_depth{}.pickle".format(skill_count_depth)
min_skill_length = 5  # Tunes the minimum number of skills a profile has to have to be included in the clustering

# Database settings
host = "mongodb://localhost:27017/"
dbname = "FutureFitAI_database"
collection_profiles = "talent_profiles"
collection_flat = "talent_profiles_flat_depth{}".format(str(skill_count_depth))
collection_encoding = "skills_integer_encoding_depth{}".format(str(skill_count_depth))
collection_counts = "skills_count_encoding_depth{}".format(str(skill_count_depth))
collection_mhot = "skill_vectors_mhot_depth{}".format(str(skill_count_depth))
remake_collection_flat = True
remake_collection_mhot = False