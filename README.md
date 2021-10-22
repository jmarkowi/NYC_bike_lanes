# Stay In Your Lane!

## Automated Bike Lane Enforcement With Neural Network Image Classification

### Author: Jesse Markowitz, October 2021


## The Problem

![a scene often seen in NYC](readme_images/cab_in_bikelane.png)

Biking is my primary mode of transportation around New York City, as it is for an increasing number of people every year. Although NYC's bike infrastructure rivals that of other cities in the country, crashes and fatalities still occur, and "nearly two-thirds of those who do not bike cite safety concerns as their main reason for not riding more often or at all." [source](https://rpa.org/work/reports/the-five-borough-bikeway) Clearly there is room to grow when it comes to making cycling in NYC safe and accessible to all. A major safety concern, and the focus of this project, is the high number of cars parked in designated bike lanes. [One study from Hunter College in 2019](http://www.hunter.cuny.edu/communications/repository/files/Bike%20Lanes%20or%20Blocked%20Lanes%20Study.pdf/) found an average of 11.5 blockages per mile of bike lane. On conventional bike lanes, the single greatest cause of these blockages was vehicular obstruction. A blocked bike lane means that cyclists are forced to weave in and out of traffic, creating a significant risk to their safety.

In my own experience biking hundreds of miles per year in the city, not a single trip goes by in which I am not forced to leave the bike lane because it is blocked by a vehicle. Most often, the vehicle is a Taxi and Limousine Commission (T&LC) car (yellow and green cabs, as well as rideshare vehicles for Uber, Lyft, etc.), a delivery truck, or a police car. The problem seems to be getting worse, even as more bike lanes are added and more people ride bicycles in NYC, yet there seems to be [little interest in enforcement](https://nyc.streetsblog.org/2021/10/21/ignored-dismissed-how-the-nypd-neglects-311-complaints-about-driver-misconduct/). **Insufficient enforcement of bike lane traffic laws creates serious safety issues for cyclists.**

![yet another cop in the bike lane](readme_images/blocked_bike_lane_nj_port_authority.png)


## Business Understanding

Over the past decade, the number of cyclings on NYC streets has grown at ever increasing rates. According to 2017's [Cycling in the City](http://www.nyc.gov/html/dot/downloads/pdf/cycling-in-the-city.pdf) report from the Department of Transportation (DOT), there were over 178 million cycling trips that year, representing an increase of over 10 million from the year before and showing a 134% growth rate in daily cycling from 2007. This increase is undoubtedly both cause and effect of the implementation of the CitiBike bike share program in 2013, as well as the hundreds of miles of bike lane infrastructure that has been added to the streets over the past several years. As of 2018, there were over 1,200 miles of bike lanes in the city, with more installed since then. The covid-19 pandemic also resulted in a [spike in bike ridership](https://www.nytimes.com/2020/03/14/nyregion/coronavirus-nyc-bike-commute.html) in NYC as commuters and city dwellers sought alternatives to crowded public transit. Commuting to work by bike accounts for approximately 20% of all trips in NYC; cycling to work in the city has grown faster than in any other major city in the country. 

![peer cities graph](readme_images/peer_cities.png)

There are four types of official signed bike routes in NYC, but only two of them are officially designated as lanes. Some are considered "protected," which means there is some sort of barrier or buffer zone between the lane and moving traffic. Most often, protected bike lanes are placed between the sidewalk and an area of parallel parking, but they may also be designated with flimsy plastic posts (often called "flexis" or "flexiposts"). Sometimes there is simply a painted buffer zone keeping traffic at bay. Protected bike lanes are also usually filled in with green paint, as well as white bike symbols.

Conventional bike lanes are located in the road, between vehicular traffic lanes and parking or sidewalks. There are no physical barriers or space, but the lanes are painted on the road with white lane lines and bike symbols. This is the type of lane the Hunter study identified as most likely to be blocked by a vehicle, which is in line with my own experience as well.

![types of bike lanes, from the NYC bike map 2021](readme_images/bike_lane_types.png)
(For the full map, visit https://www1.nyc.gov/html/dot/html/bicyclists/bikemaps.shtml)

On September 15, 2021, the NYC DOT released a ["Request for Expressions of Interest"](https://a856-cityrecord.nyc.gov/RequestDetail/20210907107) to create a system for automated bike lane enforcement. A system for bus lanes called the Automated Bus Lane Enforcement (ABLE) system was created by [Siemens Mobility](https://www.mobility.siemens.com/us/en/company/newsroom/short-news/first-ever-mobile-bus-lane-enforcement-solution-in-new-york.html) and installed in 2010 and has been expanded since then with great success, as measured by increased route speed and ridership. (In fact, the MTA plans to add hundreds more onboard and fixed cameras to bus routes to increase enforcement to 85% of all bus lanes [source](https://www1.nyc.gov/html/dot/html/bicyclists/bikemaps.shtml).) 

Automating enforcement of bike lane traffic laws would have the immediate effect of increase enforcement from what seem to be negligible levels. The T&LC enforces violators within their ranks by filing summonses for taxi and rideshare drivers in bike lanes (you can help by submitting reports through 311 easily via the [Reported NYC app](https://reportedly.weebly.com/)), but it seems that other drivers are rarely if ever held accountable for parking in the bike lane. This may be related to the fact that the police themselves are some of the worst violators, often using bike lanes as free parking for their personal and service vehicles, and [hostile to those who ask](https://nyc.streetsblog.org/2021/10/18/bad-cop-bad-cop-nypd-threatens-tipster-for-filing-311-complaints-about-illegal-parking/) them to follow the laws they are meant to enforce. Automated enforcement would also benefit the city's cyclists by reducing the need for active police involvement, especially on streets where the problem is the greatest.

Last year [at a press conference](https://www.masstransitmag.com/bus/vehicles/press-release/21119742/mta-new-york-city-transit-mta-busmounted-camera-program-begins-issuing-bus-lane-violations-on-b44-sbs-route) on another expansion of the ABLE system, Craig Cipriano, then the "acting MTA Bus Company president and senior vice president for buses of NYC Transit", said, "'Bus lanes are for buses.'" I believe that the same can and will be true for bike lanes. By making sure that bike lanes are free from cars, we can make them safer and more accessible to all.

## Purpose of Analysis

The goal of this project is to take a necessary step toward automating enforcement by building and training a model to identify when a car is parked in a bike lane. The scope of this project is limited to train a model to recognize when a vehicle is obstructing a bike lane, given that the images used for training and evaluation are guaranteed to contain a bike lane. This scope is based on the idea of an automated enforcement system comprised of stationary cameras placed on streets with bike lanes. The cameras would point up or down the street in order to monitor as much of the length of the bike lane as possible. The image below shows a speed camera used in the city, but illustrates this idea.

![speed camera](readme_images/speed_camera.png)

Models are evaluated using accuracy and precision (the true-positive rate), as well as an inspection of model decision-making using the [Lime](https://github.com/marcotcr/lime) package. Precision is used to ensure that false-positives are kept to a minimum. 


## Data & Methods

The dataset consists of just over 1,800 images of New York City bike lanes, up from about 1,600 at the beginning of the project. Just over half of these images are of a bike lane obstructed by a vehicle, which comprises the target class. The rest of the images are of bike lanes without vehicular obstruction, showing entirely empty bike lanes or, on occassion, bike lanes with cyclists or pedestrians. The small size of the dataset is one of the most significant limitations of this project.

The images in the dataset were collected from a variety of sources:
 - The [Reported app's Twitter page](https://twitter.com/Reported_NYC), which tweets all traffic violations reported through the app
 - A large dataset of images provided by [Ryan Gravener](https://github.com/snooplsm), who is working on an image recognition project for Reported
 - Screenshots from Google Maps Street View
 - Manual collection (i.e., taking photos while biking around the city--this is the source of the vast majority of the non-target images of unobstructed bike lanes)
 
### Preprocessing

Several steps were taken to prepare images for modeling. First, images collected manually (taken with a smartphone camera) had to be reoriented to ensure they were being fed into the model correctly.

![unoriented images](readme_images/unoriented_images.png)
![oriented images](readme_images/oriented_images.png)

Next, some images had to be cropped. Many images collected via Reported contain timestamps printed at the top of the image, which enhances the photo's value as evidence in a potential hearing. However, this creates a potentially confounding factor in the dataset because images with timestamps will be overrepresented in the target class. Without removing this feature, it's possible that the model will use it to predict the target class, rather than attending to real features in the image.

![uncropped images](readme_images/uncropped_images.png)
![cropped images](readme_images/cropped_images.png)

Many images were deemed unsuitable for training and were removed entirely from the dataset. Images were generally removed that:
 - did not show both lane lines of a bike lane (lines too faint or photo taken too "close up" to a vehicle)
 - showed a bike lane from the side (photo was taken from across the street, facing across the street or toward the sidewalk)
 - contained too many cyclists, motorbikes, or pedestrians such that the bike lane was significantly obstructed
 - were taken at night (these were extremely overrepresented in the target class)
 - showed a car crossing a bike lane legally (i.e., crossing an intersection) or parked in a legally ambiguous zone
 - were ambiguous (such as showing a potential vehicular obstruction too far in the distance to tell for sure)
 
The following are a small set of representative examples of the over 200 images that were ultimately removed from the orginal dataset:

![examples of unused images](readme_images/examples_unused.png)

Many of these decisions were subjective judgments and there were a surprisingly large number of images that were ambiguous. These images were kept in a separate `unused_images` folder for later inclusion or testing and as non-examples. 

Because the dataset is also limited, it was thought best to restrict the data to only the clearest examples that match the intended use with an automated enforcement system and scope of the project. The hope is that with more data and continued model training, a model can be created that also recognizes pedestrians or cyclists as well.

Finally, the image set was split into a training and holdout testing set. Later, more images were added to the corpus and another split was made for validation purposes. This resulted in the following file structure for fully processed images:

This resulted in the following file structure for processed images:

```
└── input_images
    ├── full_combined
    ├── new
    ├── test
    ├── train
    └── validation
```
Each folder of images contains 2 subfolders to designate image classes, as shown below with one example:
```
└── input_images
    ├── full_combined
    │    ├──open_bike_lane
    │    └──vehicle_bike_lane
```

### EDA

Both the validation and holdout test sets contain an even 50/50 split of image classes (50 images of each). Effort was made during data collection to keep the class balance in the training set as even as possible. Although there are slightly more target class images, this imbalance was not considered large enough to be a serious issue.

![class distribution](readme_images/class_distrib.png)

Below are samples of a few images from each class.

![open bike lane images](readme_images/open_images.png)
![vehicle obstructed bike lane images](readme_images/target_images.png)

The hope is that a neural network, and especially the pattern-detecting filters in a convolutional neural network, will be able to detect the lane lines and vehicle shapes in the images. The images are so consistent in content and perspective and similar both within and between the classes except for the main feature (a vehicle in the bike lane). Because of this, it seems reasonable that a model can predict the classes with high accuracy, even with a small dataset.

### Modeling

Over 20 models were trained on the data, starting with simple fully connected dense neural networks, progressing through convolutional neural networks, and finally ending with transfer learning models. These powerful models are built using state-of-the-art pre-trained models as a base, such as VGG-16 and InceptionV3. These models have been trained on the [ImageNet dataset](https://www.image-net.org/), a set of over 14 million images labeled and classified into 1,000 categories.

Although this project is a simple binary classification project, the idea is that these models, which are comprised of dozens of convolutional layers, have learned to recognize many, many simple and complex patterns. By using not only their architecture, but also the pre-tuned weights and hyperparameters for each layer, it's possible to harness that knowledge and apply it to a new task.

The following is a diagram showing the structure of the VGG-16 model. The three Dense layers were removed, then new Dense layers added on to apply the pre-trained model to this classification task:

![vgg16 architecture](readme_images/vgg16.png)

Image augmentation helped both to avoid overfitting and to artificially increase the size of the dataset. Image augmentation through the Keras `ImageDataGenerator` is performed randomly and on the fly. This allows the model to train on a variety of images beyond just those in the dataset. It's important to use parameters that the model is likely to see. These parameters were chosen to account for images from either side of the street, in a variety of lighting conditions, spotting a variety of vehicles in the bike lane at a variety of distances from the camera and locations in the frame. Examples of augmentations performed on a single target class image are shown below:

![augmented images](readme_images/image_augmentation.png)

## Results

The final best model chosen for the task has as its base the VGG-16 pre-trained model architecture, topped with several fully connected dense layers. On a validation set, it performed with 94% accuracy and 100% precision, although as the validation set contained only 100 images, these metrics were unlikely to remain high on previously unseen data. Sure enough, on the holdout test set, also only 100 images, the model predicted with only 91% accuracy and 91% precision. It is possible that the model is still overfit to the data and would benefit from more regularization strategies, such as l2 regularization or dropout layers. When these techniques were tried on their own, they caused a drop across all metrics, but this could be desireable if it means more consistency in the long run on new images.

![final confusion matrix](readme_images/final_conf_matrix.png)

Performing above 90% is a great start for a small dataset. I am confident that increasing the size of the dataset to at least 1,000 images per class, as well as trying out more strategies and iterations on the transfer learning model, will result in improved performance on unseen data. Casual testing on nighttime images, which were not used in training, resulted in over 80% accuracy overall and 97% precision on an imbalanced dataset (heavily in favor of the target class). This suggests that the model is generalizable and supports the idea that more training on more data will help it. 

Ultimately, the use case stated for this model would allow for continued training and improvement over time as more images are collected.


## Conclusion

### Recommendations

New York City needs automated bike lane enforcement. The bike lanes are too often treated as free parking for the city's drivers, especially by delivery vehicles, taxis, and police vehicles. This causes dangerous conditions for the ever-increasing number of cyclists on the streets who depend on bike lanes to provide a safer corridor, free from traffic. When vehicles are stopped in the bike lane, it forces cyclists to merge into traffic and weave around cars, putting them at risk of fatal injury.

Automated enforcement would increase the efficiency and consistenty of ticketing, as well as reduce the need for police to physically engage with drivers. This would save time and human resources, and likely save money as well. I recommend starting with stationary cameras, pointing down bike lanes on longer, straight streets that are largely free from other obstructions. Historic data from Reported could be used to identify and locate especially problematic areas in which bike lanes are consistently clogged. (I can think of a half a dozen locations around the city that have bike lanes I have never been able to actually ride down due to parked vehicles blocking the lanes.)

This is a manageable, if not preventable issue. The Department of Transportation reported that installing the ABLE system on bus lines increased bus route speeds and ridership; they are working to expand the system to cover over 85% of all NYC bus routes. Creating an analogous system for bike lanes would increase safety for bike commuters and anyone else who cycles in the city.

### Possible Next Steps

To improve the model, the best and most important next step is to collect more data. Beyond that, there are several possible avenues to take this:
 - Add additional classes (by adding more data), including identifying images with bikes, motorbikes, or other types of vehicles
 - Move to object *detection* rather than just image classification to identify and locate vehicles in the image
 - Incorporate Automatic License Plate Recognition (APLR) for automated ticketing and look into options for connecting for existing enforcement systems, like ABLE or red light/speed cameras in the city
 - Connect my model to the Reported app to assist in its development

## For More Information

See the full analysis in the [Jupyter Notebook](nyc_bike_lanes.ipynb) or review this [presentation](nyc_bike_lanes_presentation.pdf)

### Structure of Repository:

```
├── code (working notebooks, named by stage of project)
├── input_images (dataset used in training models)
├── models (saved .h5 files of trained models and pickled training histories)
├── other images (unused and not fully processed images)
├── nyc_bike_lanes.ipynb
├── nyc_bike_lanes_presentation.pdf
├── functions.py (custom functions)
├── model_tracker.csv
└── README.md
```
