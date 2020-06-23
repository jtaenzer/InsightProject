# ClusTitle

Try the webapp: [link](www.machinelearningprojects.xyz)

## Table of contents

* [Overview](https://github.com/jtaenzer/InsightProject#overview)
* [Usage](https://github.com/jtaenzer/InsightProject#usage)
* [Analysis](https://github.com/jtaenzer/InsightProject#analysis)
* [Classify](https://github.com/jtaenzer/InsightProject#classify)

## Overview

The goal of this project is to take ~1m unique, self reported job titles in ~300M online talent projects and reduce the
taxonomy of job titles using clustering.

Therefore the first, very reasonable, assumption is that there exists some embedding of features in the online talent
profiles where profiles containing similar job titles will be closer together. If this assumption is not correct, it
will not be possible to obtain coherent clusters from the dataset.

Potential features of interest from the online talent profiles:

* skills
* past job titles / past job experience
* education level
* job title contents (senior, junior, etc)

In its current state, this code only tries to cluster using skills, but extending it to other features should be 
straightforward. The second, less reasonable, assumption is that all of the skills listed in each profile can be used to
describe the most recently reported job title. This will often be incorrect as people will make career transitions and
learn new skills along the way, and are unlikely to remove skills irrelevant to their current job from their profile.

Finding a way to identify which skills are associated to the current job in a profile could potentially be very helpful.

# Usage

## Rebuilding and cleaning the DB

Re-building the DB generally shouldn't be necessary. The build_db.py script assumes that correctly formatted jsons are
available.

Before running the script, note that the DB settings and data path are hard coded so it should be opened and edited as
necessary.

```
python scripts/build_db.py
```

Cleaning the DB via the clean_db.py script will remove symbols and extra white space in the job title fields and then 
map common acroynyms and abbreviations to reduce redundancy in the job title space. A new field called "clean_title" is
created. 

*Titles still containing non alphanumeric characters after common symbols are removed are dropped in an attempt
drop titles written in foreign languages. This cleaning step may not be desirable!*

Before running the script, note that the DB settings and data path are hard coded so it should be opened and edited as
necessary.

```
python scripts/clean_db.py
```

## Clustering

Before running any clustering, it is strongly advised that you open src/configs/cluster_config.py and read the
descriptions of the tunable parameters. Clustering can be extremely memory intense so appropriate hyperparameter
choices are a must. For example, if after the pipeline you have a matrix of size 200,000 x N entering the clustering,
around 380 GB of RAM will be necessary to run the clustering. Depending on the size of the initial database, it may take
some experimentation to find appropriate parameters.

Running the clustering:

```
python src/cluster.py
```

Most of the heavy lifting takes place in the Pipeline class in pipeline.py, which is thoroughly commented.

## Analysis

A script to perform a simple analysis of the clustering output is provided in src/cluster_analysis.py. Before running
the script it is advised to open src/configs/analysis_config.py and read the descriptions of the tunable parameters.  

Running the analysis:

```
python src/clustering_analysis.py
```

This will find some number of clusters and label them based on the most frequent title appearing in the cluster.
Histograms and word clouds of the titles and skills appearing in each cluster can be generated as well. The centroid
of each cluster will be calculated and saved in a binary.

### Existing Binaries

Binaries for a 120k and 200k profile clustering are in the binary folder in this repo. The analysis can be run directly 
on these binaries. See:

```
binaries/120k_profiles
binaries/200k_profiles
```

## Classify

This is still a work in progress but the goal is to build a classifier from the clustering model. Starts similar to 
clustering_analysis by finding pure clusters based on the parameters in analysis_config.py. The clusters are labeled
by the most frequently appearing title in the cluster, and these can be used to create a supervised learning model.
Currently sklearn's MLPClassifier is used but this may not be the ideal model.

Note that since the input matrix to the clustering is rebuilt in full, running this script may involve large memory
allocations.

Training the classifier:

```
python src/classify.py
```