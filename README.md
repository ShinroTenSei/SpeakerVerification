This is a speaker identification project. 

Data set is based on Voxceleb1 with over 10k uttrance.

Attention mechanism and triplet loss are applied in this project.

In predefined package, following functions are defined:

data_pipeline: the module is created to transform origin .wav signals(which can be recorded also) into speaker features in the form of rolling windows.

models: this module contains major component of predefined dnn models, classes corresponds to different models which contains a builder to set up graph for itself.	
	returns an object with tensors needed of the computation graph. 

trainer: this module contains functions to train, predict, finetune, save, load models. 
	 Fed with tensors, dataframe. 


train: python control module to run the whole thing.
	1. load and prepare data first
	2. setup the models by assign a model class to a variable
	3. pass the returned variables into a trainer to conduct training operations.
	4. if early stopping is triggered by the trainer, stop, load best model, lower the lr and fine tune.
	5. repeat 4 until the model gets converged.

to do:

Redefine function load data since duplicate load_data exists.

	

# SpeakerVerification
