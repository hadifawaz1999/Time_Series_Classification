import numpy as np

#Hello this is a program to classify time series using 3 methods that
#we will mention later and find the accuracyof our work or to find
#the SAX representation of a time series or image recognition ny logistic
#recognition with Neural Network mindset.


#You have two choices then:")
#     Choose 1 to classify time series.
#     Choose 2 to find SAX representation.
#     Choose 3 to classify images.
#     Choose 4 to check explanation of classifying by Logistic Regression

#What is your choice:
choice_of_work=4

#####if your choice of work was 1:

#Give file name:
file_name="ProximalPhalanxOutlineAgeGroup"
#What method do you want:
#         Choose 1 for euclidean distance.
#         Choose 2 for dtw without warping window.
#         Choose 3 for dtw with warping window.

#Your choice is
choice=1
#if your choice was 3:
#What is the adjustment window:
w=1
#NB: if your choice was not 3

#####if your choice of work was 2:

#What is the number of PAA segments you want to test this on?
p=80
#How many characters from the alphabet you wana use?
alpha=1

####if your choice of work was 4:
file="Coffee"