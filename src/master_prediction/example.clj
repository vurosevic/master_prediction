(ns ^{:author "Vladimir Urosevic"}
master-prediction.example
  (:require [uncomplicate.neanderthal.core :refer :all]
            [uncomplicate.neanderthal.vect-math :refer :all]
            [uncomplicate.neanderthal.native :refer :all]
            [uncomplicate.neanderthal.linalg :refer :all]
            [criterium.core :refer :all]
            [clojure.string :as string]
            [clojure.core :as core]
            [master-prediction.data :refer :all]
            [master-prediction.neuralnetwork :refer :all]
            ))

;; snimanje i ucitavanje konfiguracije
(def mreza-nt (atom (create-network-from-file "nn-mreza-122.csv")))
(save-network-to-file @mreza-nn "nn-mreza-126.csv")

(-> @mreza-nt)

;; predikcija za specificirani ulazni vektor
(def input-test [2010 9 25 7 0 3424 3060 2861 2772 2761 2971 3435 4015 4195 4261 4215
                 4268 4225 4161 4047 4003 3995 3992 4365 4956 4849 4670 4273 3848 2761
                 4956 93622 14 19.5 25 15.96 14 19.27 25 15.82 23.27 48.95 1011.73 3452
                 3104 2886 2794 2757 2890 3084 3617 4017 28601 17 18.59090909 22 17.80976667
                 17 18.59090909 22 17.62029836 28.68181818 68 1002.318182 1])

(def input-test2 [2009 1 26 2 0 5113 4732 4399 4150 4004 4011 4186 4419 4852 5221 5473 5650
                  5648 5636 5492 5475 5657 5970 5960 6038 6007 5872 5695 5361 4004 6038 125021
                  2 3.3 4 5.08 -1 -0.48 1 3.31 16.17 100 996.96 5100 4730 4380 4234 4090 4292
                  4931 5600 5884 43241 3 4.608695652 8 4.098109931 0 2.565217391 8 1.292125826
                  8.086956522 89.95652174 1005 1])

(def input-test3 [2015 4 24 6 0 4128 3718 3392 3275 3284 3403 3909 4526 4600 4645 4553 4424 4393
                  4310 4206 4167 4101 4092 4245 4687 5230 5168 4888 4462 3275 5230 101806 10 14.15
                  9 12.39 11 13.66 9 11.7 8.83 49.13 1018.7 4025 3572 3296 3160 3161 3322 3762
                  4346 4578 33222 10 16.61702128 9 13.32326699 10 16.34042553 9 12.75154427
                  8.212765957 50.21276596 1015.765957 1])

(def norm-test (atom (fge 65 1)))
(def norm-in (normalize-input-vector input-test input-trainig-dataset @norm-test))
(def norm-in2 (normalize-input-vector input-test2 input-trainig-dataset @norm-test))
(def norm-in3 (normalize-input-vector input-test3 input-trainig-dataset @norm-test))

(def temp-variables4 (atom (create-temp-record @mreza-nn norm-in)))
(predict @mreza-nn norm-in @temp-variables4)
(restore-output-vector target-trainig-dataset (predict @mreza-nn norm-in @temp-variables4) 0)
(entry (restore-output-vector target-trainig-dataset (predict @mreza-nn norm-in @temp-variables4) 0) 0)

;; kreiranje i treniranje mreze
(def mreza-nn (atom (create-network 64 [80 200 80] 1)))
;; (xavier-initialization-update @mreza-nn)
;; (set-biases-value @mreza-nn 0)

(def temp-variables3 (atom (create-temp-record @mreza-nn (:normalized-matrix input-test-dataset))))

(time (train-network @mreza-nn (:normalized-matrix input-trainig-dataset)
                               (:normalized-matrix target-trainig-dataset) 100 4
                               0.015557 0.9))

(-> input-trainig-dataset)

(nth (:layers @mreza-nt) 0)
(get-bias-vector (nth (:layers @mreza-nt) 0))
(submatrix (nth (:layers @mreza-nt) 0) 0 1 80 1)

(nth (:layers @mreza-nn) 0)
(get-bias-vector (nth (:layers @mreza-nn) 0))
(submatrix (nth (:layers @mreza-nn) 0) 0 1 80 1)

(mul
  (nth (:temp-vector-vector-h-gradients @temp-variables3) 0)
  (get-bias-vector (nth (:layers @mreza-nt) 0))
  )

(copy (get-bias-vector (nth (:layers @mreza-nt) 0)))
(get-bias-vector (nth (:layers @mreza-nt) 0))
(get-bias-vector (nth (:layers @mreza-nn) 0))

(mul
  (nth (:temp-vector-vector-h-gradients @temp-variables3) 0)
  (copy (get-bias-vector (nth (:layers @mreza-nn) 0)))
  )

(predict @mreza-nn (:normalized-matrix input-test-dataset) @temp-variables3)

;; evaluacija mreze
(evaluate-original-mape
  (evaluate-original
    (restore-output-vector target-test-dataset (predict @mreza-nn (:normalized-matrix input-test-dataset) @temp-variables3) 0)
    (restore-output-vector target-test-dataset (:normalized-matrix target-test-dataset) 0)
    ))

(evaluate-original
  (restore-output-vector target-test-dataset (predict @mreza-nn (:normalized-matrix input-test-dataset) @temp-variables3) 0)
  (restore-output-vector target-test-dataset (:normalized-matrix target-test-dataset) 0)
  )

(restore-output-vector target-test-dataset (predict @mreza-nn (:normalized-matrix input-test-dataset) @temp-variables3) 0)
(restore-output-vector target-test-dataset (:normalized-matrix target-test-dataset) 0)

(def nn (evaluate-original
          (restore-output-vector target-test-dataset (predict @mreza-nn (:normalized-matrix input-test-dataset) @temp-variables3) 0)
          (restore-output-vector target-test-dataset (:normalized-matrix target-test-dataset) 0)
          ))

(predict @mreza-nn (:normalized-matrix input-test-dataset) (create-temp-record @mreza-nn (:normalized-matrix input-test-dataset)))

(feed-forward @mreza-nn (:normalized-matrix input-test-dataset) @temp-variables3)
(:layers-output @temp-variables3)

(def temp-variables2 (atom (create-temp-record @mreza-nn (:normalized-matrix input-trainig-dataset))))
(quick-bench
  (learning-once @mreza-nn (:normalized-matrix input-trainig-dataset) 4 (:normalized-matrix target-trainig-dataset)
                   @temp-variables2 0.015 0.4)
  )

(read-data-from-csv "resources/data_prediction.csv")

(:layers @mreza-nn)
(:layers @mreza-nt)

(get-bias-vector (nth (:layers @mreza-nn) 4))
(get-bias-vector (nth (:layers @mreza-nt) 4))