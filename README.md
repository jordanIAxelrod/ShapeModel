# Implementation of IMCP (Iterative Median Closest Point) aglorithm found in Performing Accurate Joint Kinematics From 3-D 

This repository contains the code to replicate [An Automated Statistical Shape Model Developmental Pipeline: Application
to the Human Scapula and Humerus](https://ieeexplore.ieee.org/document/6949148). This model is an improvement over 
standard Iterative Closest point methods in that it uses median statistics to create a Consensus shape.

## Installation
This project can be installed through pip by cloning this repository
```bash
pip install git+https://github.com/jordanIAxelrod/ShapeModel
```

## Usage
The main class contained in this repository is the ShapeModel. It takes a set of object outlines as numpy arrays and 
fits a shape model to them. It can then fit new shapes and summarize them as a smaller vector. 

Please see example.ipynb for details of its use.

## Explanation 
To create a shape model, instances of the shape must be aligned and the 
points matched inorder to run PCA on the shapes. This algorithm matches 
the points to the median of the closest points in the other shapes to eliminate the 
bias found in choosing a "root shape."

This has only been tested on cubes. The variation along any axis of a cube is very
similar. Futher testing on the bone segmentations once they are ready will be needed
to show correctness.

TestShapeRegistration.py generates these cubes and runs IMCP and prints the results.

The initial cubes are randomly rotated and translated. As shown in figure 1

![Initial Cubes](img/Initial%20cubes.png "Initial Cubes")

The model then outputs the points, their rigid transformations with respect to the
median shape, and a membership map.

The final cubes are shown in figure two

![Final Cubes](img/Point%20Cloud%20End.png "Ending Point Clouds")

The median shape is also found

![Median Shape](img/MedianShape.png "Median Shape")

We suspect these results will be better with bone shapes. It would also
be nice to devise a measure of goodness of fit. May involve making point clouds 
where the correspondance between points is known a priori.

## Authors

Jordan Axelrod.

With acknowledgements to Hanxue Gu and Dr. Maciej Mazurowski
