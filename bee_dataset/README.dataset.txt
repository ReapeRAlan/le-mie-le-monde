# Honey Bee Detection Model > 883base-x7aug
https://universe.roboflow.com/matt-nudi/honey-bee-detection-model-zgjnb

Provided by a Roboflow user
License: CC BY 4.0

# Purpose of  the Project
This project started as a way to add real-time counts of bees with/without pollen entering my backyard beehive to append some additional information to a livestream of the hive, and to correlate behavior at the hive entrance to weather, temperature, etc. Since then, I've added additional training data not specific to my hive which accounts for classification of drones and queens in addition to bees/pollen bees. Currently the model generalizes reasonably well, but more training data is required. 

# Assessed Classes & Labeling Guidelines

* bees (either workers or foragers)
* bees carrying pollen
* drones
* queens

Labeling should cover the entire body of the bee, **excluding** the wings as per the following example:
![](https://i.imgur.com/DCaj9ir.png)

For the class of bees carrying pollen, it is acceptable to extend the box to include the visible pollen packs to distinguish this from the bee class:
![](https://i.imgur.com/pWhuznH.png)

# Sample Results

Sample video of backyard hive entrance with low to moderate level of activity:
[https://www.youtube.com/watch?v=qZW5eYd0Yw8&t=2266s](https://www.youtube.com/watch?v=qZW5eYd0Yw8&t=2266s)

Generic sample video of single bee:
[https://www.youtube.com/watch?v=A1x6VA8TWCg](https://www.youtube.com/watch?v=A1x6VA8TWCg)

# Latest YOLOv5 Weights Files
Latest weights files for use by others are posted on my github here: [https://github.com/mattnudi/bee-detection](https://github.com/mattnudi/bee-detection)

These files will be updated as more images are added to the dataset