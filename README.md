# JobTitleClustering

The goal of this project is to take ~1m unique, self reported job titles in ~300M online talent projects and reduce the taxonomy of job titles using clustering.

Therefore the first, very reasonable, assumption is that there exists some embedding of features in the online talent profiles where profiles containing similar job titles will be closer together. 
If this assumption is not correct, it will not be possible to obtain coherent clusters from the dataset.

Potential features of interest from the online talent profiles:

* skills
* past job titles / past job experience
* education level
* job title contents (senior, junior, etc)

In its current state, this code only tries to cluster using skills, but extending it to other features should be straightforward. 
The second, less reasonable, assumption is that all of the skills listed in each profile can be used to describe the most recently reported job title.
This will often be incorrect as people will make career transitions and learn new skills along the way, and are unlikely to remove skills irrelevant to their current job from their profile.

Finding a way to identify which skills are associated to the current job in a profile could potentially be very helpful.

## Rebuilding and cleaning the DB



