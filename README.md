# Setup for algorithms
Each algorithm needs to have the following setup to work with the main file.
1. The class inside the algorithm file needs to be named ```Algorithm```
2. The method used to start the algorithm needs to be called ```run``` with no input parameters (besides self)
3. There should be a method outside the Algorithm class that is named ```get_params_gs```
    * This method will return a product of all possible hyperparameter values used for Grid Search
    * It should not take any input
    * Needs to return an object of type itertools.product
4. The ```Algorithm``` class needs to be pickleable
    * NOTE: You may not have to do anything to make the class pickleable. All that is required is that no unpickleable objects or modules are used or imported within the class
    * IF an unpickleable object or module is required, then look at genetic algorithm file to see how to make the class pickleable. It requires the methods ```__getstate__``` and ```__setstate__```
5. Have a method outside the ```Algorithm``` class called ```to_string``` that takes in no parameters. It will return the string that is the name of the algorithm to put on output files.
