In the demo file, everything is in the same file, which is bad format.
So separate them into different translation units
the data set processing is in a data.py file
the model related is in a model.py file, create a model class, since I may warp the base model for other modification. also I may define my own loss function in the future
the training and inference in 2 different units, train.py, inference.py
write the program to the save dir of this file.
