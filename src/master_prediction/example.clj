(ns ^{:author "Vladimir Urosevic"}
master-prediction.example
  (:require [uncomplicate.neanderthal.core :refer :all]
            [uncomplicate.neanderthal.vect-math :refer :all]
            [uncomplicate.neanderthal.native :refer :all]
            [uncomplicate.neanderthal.linalg :refer :all]
            [criterium.core :refer :all]
            [clojure.string :as string]
            [clojure.core :as core]
            [clojure.java.io :as io]
            [master-prediction.data :refer :all]
            [master-prediction.neuralnetwork :refer :all]
            ))

;; loading and saving neural network
(def mreza-nn (atom (create-network-from-file "early-stopping-net-10-100.csv")))
(save-network-to-file @mreza-nn "nn-net-123.csv")

;; prediction for specified input vector
(def input-test [2010 9 25 7 0 3424 3060 2861 2772 2761 2971 3435 4015 4195 4261 4215
                 4268 4225 4161 4047 4003 3995 3992 4365 4956 4849 4670 4273 3848 2761
                 4956 93622 14 19.5 25 15.96 14 19.27 25 15.82 23.27 48.95 1011.73 3452
                 3104 2886 2794 2757 2890 3084 3617 4017 28601 17 18.59090909 22 17.80976667
                 17 18.59090909 22 17.62029836 28.68181818 68 1002.318182])

(def input-test2 [2009 1 26 2 0 5113 4732 4399 4150 4004 4011 4186 4419 4852 5221 5473 5650
                  5648 5636 5492 5475 5657 5970 5960 6038 6007 5872 5695 5361 4004 6038 125021
                  2 3.3 4 5.08 -1 -0.48 1 3.31 16.17 100 996.96 5100 4730 4380 4234 4090 4292
                  4931 5600 5884 43241 3 4.608695652 8 4.098109931 0 2.565217391 8 1.292125826
                  8.086956522 89.95652174 1005])

(def input-test3 [2015 4 24 6 0 4128 3718 3392 3275 3284 3403 3909 4526 4600 4645 4553 4424 4393
                  4310 4206 4167 4101 4092 4245 4687 5230 5168 4888 4462 3275 5230 101806 10 14.15
                  9 12.39 11 13.66 9 11.7 8.83 49.13 1018.7 4025 3572 3296 3160 3161 3322 3762
                  4346 4578 33222 10 16.61702128 9 13.32326699 10 16.34042553 9 12.75154427
                  8.212765957 50.21276596 1015.765957])

(def norm-in (normalize-input-vector input-test input-trainig-dataset))
(def norm-in2 (normalize-input-vector input-test2 input-trainig-dataset))
(def norm-in3 (normalize-input-vector input-test3 input-trainig-dataset))

(def temp-variables (atom (create-temp-record @mreza-nn norm-in)))
(predict @mreza-nn norm-in @temp-variables)

(restore-output-vector target-trainig-dataset (predict @mreza-nn norm-in @temp-variables))
(entry (restore-output-vector target-trainig-dataset (predict @mreza-nn norm-in @temp-variables)) 0)

(with-progress-reporting
(quick-bench
  (predict @mreza-nn norm-in @temp-variables)
  )
)

(with-progress-reporting
  (quick-bench
    (restore-output-vector target-trainig-dataset (predict @mreza-nn norm-in @temp-variables))
    )
  )

;; creating and training neural network

(def mreza-nn (atom (create-network 64 [100 200 200 100] 1)))
(def mreza-nn (atom (create-network 64 [100 200 100] 1)))
(def mreza-nn (atom (create-network 64 [200 200] 1)))
(def mreza-nn (atom (create-network 64 [100 100] 1)))
(def mreza-nn (atom (create-network 64 [100] 1)))

(def temp-variables3 (atom (create-temp-record @mreza-nn (:normalized-matrix input-test-dataset))))

(time (train-network @mreza-nn (:normalized-matrix input-trainig-dataset)
                     (:normalized-matrix target-trainig-dataset) 500 10
                     0.0157 0.9))
;; 0.00357
;; 0.0157
;; 0.00557
;; 0.01057

(with-progress-reporting
     (quick-bench (train-network @mreza-nn (:normalized-matrix input-trainig-dataset)
                                 (:normalized-matrix target-trainig-dataset) 1 20
                                 0.01708557 0.9))
     )

(def input-test [2010 9 25 7 0 3424 3060 2861 2772 2761 2971 3435 4015 4195 4261 4215
                 4268 4225 4161 4047 4003 3995 3992 4365 4956 4849 4670 4273 3848 2761
                 4956 93622 14 19.5 25 15.96 14 19.27 25 15.82 23.27 48.95 1011.73 3452
                 3104 2886 2794 2757 2890 3084 3617 4017 28601 17 18.59090909 22 17.80976667
                 17 18.59090909 22 17.62029836 28.68181818 68 1002.318182])

(def norm-in (normalize-input-vector input-test input-trainig-dataset))
(def temp-variables (atom (create-temp-record @mreza-nn norm-in)))

(with-progress-reporting
(quick-bench
  (predict @mreza-nn norm-in @temp-variables)
  )
)

;; evaluation
(evaluate-mape
  (evaluate
    (restore-output-vector target-test-dataset (predict @mreza-nn (:normalized-matrix input-test-dataset) @temp-variables3))
    (restore-output-vector target-test-dataset (:normalized-matrix target-test-dataset))
    ))

(evaluate
  (restore-output-vector target-test-dataset (predict @mreza-nn (:normalized-matrix input-test-dataset) @temp-variables3))
  (restore-output-vector target-test-dataset (:normalized-matrix target-test-dataset))
  )

;; create file for drawing diagram of convergence
(create-predict-file @mreza-nn (:normalized-matrix input-test-dataset)
                     target-test-dataset "test-file.csv")
