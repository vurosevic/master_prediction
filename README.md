# Implementation of neural networks for predicting the consumption of electricity using the programming language of Clojure

### Master thesis

## The first implementation - source code

## Usage

#### Create new neural network

> ;; we create new neural network with 64 inputs, 2x100 neurons by hidden layers and one neuron in output layer

> (def net-nn (atom (create-network 64 [100 100] 1)))

#### Train network
> ;; train network with input matrix / target matrix

> ;; 500 is number of epochs,

> ;; 10 is size of minibatch,

> ;; 0.0015 is speed learning

> ;; 0.9 is momentum coeficient

> (train-network @net-nn (:normalized-matrix input-trainig-dataset)
                       (:normalized-matrix target-trainig-dataset) 500 10
                       0.0015 0.9)

#### Network evaluation

> ;; Evaluation by MAPE metric

> ;; first we must to create temp variables

> (def temp-var (atom (create-temp-record @net-nn (:normalized-matrix input-test-dataset))))

> ;; next, we can do evaluation

> (evaluate-mape
    (evaluate
      (restore-output-vector target-test-dataset (predict @mreza-nn (:normalized-matrix input-test-dataset) @temp-var))
      (restore-output-vector target-test-dataset (:normalized-matrix target-test-dataset))
      ))

#### How to use network

> ;; first, we must prepate input vector and temp variables

> (def input-test [2010 9 25 7 0 3424 3060 2861 2772 2761 2971 3435 4015 4195 4261 4215
                 4268 4225 4161 4047 4003 3995 3992 4365 4956 4849 4670 4273 3848 2761
                 4956 93622 14 19.5 25 15.96 14 19.27 25 15.82 23.27 48.95 1011.73 3452
                 3104 2886 2794 2757 2890 3084 3617 4017 28601 17 18.59090909 22 17.80976667
                 17 18.59090909 22 17.62029836 28.68181818 68 1002.318182])

> (def norm-in (normalize-input-vector input-test input-trainig-dataset))

> (def temp-variables (atom (create-temp-record @net norm-in)))

> ;; now we can predict, but result is normalized

> (predict @net-nn norm-in @temp-variables)

> ;; if we can see real value, we must restore value

> (restore-output-vector target-trainig-dataset (predict @net-nn norm-in @temp-variables))

#### Save state in file

> ;; when your network good trained, you can save state in file.

> (save-network-to-file @net-nn "nn-net.csv")

#### Load network from file

> ;; create network from file with filename "nn-net.csv"

> (def new-nn (atom (create-network-from-file "nn-net.csv")))

## License

Copyright © 2018 Vladimir Urosevic

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
