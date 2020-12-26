# Visualize Machine Learning metrics with Tensorflow and Tensorboard

An important step in understanding machine learning processes and evaluations is the interpretation of various metrics.

It can be very time-consuming to read out individual results numerically, especially if the model has been trained for many epochs. 
This has already cost me some precious time and nerves in the past. Fortunately, there are some tools out there that can help. 
In the following example I would like to show you an application with which both the architecture and some important metrics of underlying algorithms are shown. 
I hope it can be of use to some of you.
The original post can be found [on Medium](https://ahadzalic.medium.com/visualize-machine-learning-metrics-with-tensorflow-and-tensorboard-6928db082830)

![example](https://github.com/hadze/machinelearning/blob/master/tutorials/tensorboard/doc/results.png =250x)

# Example files for displaying results in Tensorboard

In the logger folder there are some files which were generated during a training process of a sample dataset. You can use them to be displayed in Tensorboard. In order to do so, just start "tensorboard -logdir=LOGDIR" in any console or terminal program –> and enter localhost 127.0.0.1:6006 on your the browser - ideally on Chrome, since it offers the most stable view.
