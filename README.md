# ReinforcementLearning-MVC

This is an invariant implementation of the algorithm from **Learning Combinatorial Optimization Algorithms over Graphs**.

In the paper, the main idea is to build up a neural network which incorporates three parts
* Structure2Vector (s2v) to embed graphs.
* Predict the values of every node.
* Use RL to auto predict and collect training data.

In this invariant implementation, some parts are replaced,
* Structure2Vector (s2v) parameters will not be trained for speeding up the computation.
* n-steps training RL is kept, yet fitted RL are not used.

The code will run smaller graphs with 4-7 nodes for 20 times and total of 100 graphs.
During the running, the embedded nodes will gain its reward through an untrained neural network.

After the data are collected, the model can be trained.
The train model will be applied to predict larger cases which with over 15 nodes.
Through the test data, we can see 
* This algorithm is capable of applying to larger cases.
* he algorithm searchs the solutions according to the degrees of nodes.
