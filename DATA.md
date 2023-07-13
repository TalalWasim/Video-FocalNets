# Dataset Preparation

We provide our labels in the `labels` directory.

## Kinetics

1. We download the official version of Kinetics from the [dataset provider](https://github.com/cvdfoundation/kinetics-dataset) and videos are resized using code [here](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/kinetics).

2. After all the videos were downloaded, prepare the csv files for training and validation as `train.csv` and `val.csv. The format of the CSV file is:

```
<path_1>,<label_1>
<path_2>,<label_2>
...
<path_n>,<label_n>
```
where `<path_i>` points to a video file, and `<label_i>` is an integer between `0` and `num_classes - 1`.

> Since some videos may no longer be available, it will lead to a small performance gap.


## Something-Something-v2
1. Please download the dataset and annotations from the [dataset provider](https://developer.qualcomm.com/software/ai-datasets/something-something).

2. Set up the training and validation CSV files as above for Kinetics.

## Diving-48

1. Please download the dataset and annotations from the [dataset provider](http://www.svcl.ucsd.edu/projects/resound/dataset.html). Note that we use the V2 splits.

2. Set up the training and validation CSV files as above for Kinetics.

## ActivityNet-v1.3

1. Please request access for the full [v1.3](http://activity-net.org/download.html) dataset [here](https://docs.google.com/forms/d/e/1FAIpQLSeKaFq9ZfcmZ7W0B0PbEhfbTHY41GeEgwsa7WobJgGUhn4DTQ/viewform). Download the entire dataset including all missing videos.

2. Download the annotations JSON file [here](http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/files/activity_net.v1-3.min.json)

3. Set up the training and validation CSV files as above for Kinetics.
