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

(def mreza-nn (atom (create-network-gaussian 64 [80 80 80 80] 1)))
(def mreza-nn (atom (create-network-gaussian 64 [80 80 80] 1)))
(def mreza-nn (atom (create-network-gaussian 64 [80 80] 1)))
(def temp-variables2 (atom (create-temp-record @mreza-nn input-730-b)))
(xavier-initialization-update @mreza-nn)

(set-biases-to-one @mreza-nn 0)

(feed-forward @mreza-nn (:normalized-matrix norm-input-730) @temp-variables2)
(predict @mreza-nn (:normalized-matrix norm-input-730) @temp-variables2)
(restore-output-vector norm-target-730 (predict @mreza-nn (:normalized-matrix norm-input-730) @temp-variables2) 0)

(def temp-variables3 (atom (create-temp-record @mreza-nn (:normalized-matrix test-norm-input-310))))
(predict @mreza-nn (:normalized-matrix test-norm-input-310) @temp-variables3)

;; reseno !!!
(mm
  (gb native-float (first (:temp-vector-vector-h-gradients @temp-variables2)))
  (trans (first (:temp-vector-vector-h-gradients @temp-variables2)))
  )

(for [x (range 1000)]
  (backpropagation @mreza-nn (:normalized-matrix norm-input-730) 0 (:normalized-matrix norm-target-730)
                   @temp-variables2 0.015 0)
  )

(get-bias-vector (nth (:layers @mreza-nn) 0))
(trans (get-weights-matrix (nth (:layers @mreza-nn) 0)) )

(backpropagation @mreza-nn (:normalized-matrix norm-input-730) 1 (:normalized-matrix norm-target-730)
                 @temp-variables2 0.015 0.4)

(with-progress-reporting
  (quick-bench
    (backpropagation @mreza-nn (:normalized-matrix norm-input-730) 4 (:normalized-matrix norm-target-730)
                     @temp-variables2 0.015 0.4)
    )
  )

(:layers-output @temp-variables2)
(submatrix (:normalized-matrix norm-input-730) 0 1 (inc (first (:config @mreza-nn))) 1)

(count (conj (:layers-output @temp-variables2)
             (submatrix (:normalized-matrix norm-input-730) 0 1 (inc (first (:config @mreza-nn))) 1))
       )

(time (train-network @mreza-nn (:normalized-matrix norm-input-730) (:normalized-matrix norm-target-730) 50
                     @temp-variables2 0.00137 0.9))

(evaluate-original-mape
  (evaluate-original
    (restore-output-vector test-norm-target-310 (predict @mreza-nn (:normalized-matrix test-norm-input-310) @temp-variables3) 0)
    (restore-output-vector test-norm-target-310 (:normalized-matrix test-norm-target-310) 0)
    )
  )

(restore-output-vector test-norm-target-310 (predict @mreza-nn (:normalized-matrix test-norm-input-310) @temp-variables3) 0)
(restore-output-vector test-norm-target-310 (:normalized-matrix test-norm-target-310) 0)

(evaluate-original
  (restore-output-vector test-norm-target-310 (predict @mreza-nn (:normalized-matrix test-norm-input-310) @temp-variables3) 0)
  (restore-output-vector test-norm-target-310 (:normalized-matrix test-norm-target-310) 0)
  )

(restore-output-vector test-norm-target-310 (:normalized-matrix test-norm-target-310) 0)
(restore-output-vector test-norm-target-310 (predict @mreza-nn (:normalized-matrix test-norm-input-310) @temp-variables3) 0)
(predict @mreza-nn (:normalized-matrix test-norm-input-310) @temp-variables3)

(def input-row-test (submatrix (:normalized-matrix norm-input-730) 0 0 (inc (first (:config @mreza-nn))) 1))
(def temp-variables2 (atom (create-temp-record @mreza-nn input-row-test)))
(feed-forward @mreza-nn input-row-test @temp-variables2)

(restore-output-vector test-norm-target-310 (predict @mreza-nn (:normalized-matrix test-norm-input-310) @temp-variables3) 0)

(first (:config @mreza-nn))
(mrows (:normalized-matrix norm-input-730))

(:restore-coeficients norm-input-730)

(mrows (:restore-coeficients norm-input-730))

(:restore-coeficients norm-input-730)

;; snimanje i ucitavanje konfiguracije
(save-network-to-file @mreza-nn "nn-mreza-136.csv")

(def mreza-tn (atom (create-network-from-file "nn-mreza-149.csv")))
(-> @mreza-tn)
(-> @mreza-nn)

(predict @mreza-tn (:normalized-matrix test-norm-input-310) @temp-variables3)

(def temp-variables3 (atom (create-temp-record @mreza-tn (:normalized-matrix test-norm-input-310))))
(evaluate-original-mape
  (evaluate-original
    (restore-output-vector test-norm-target-310 (predict @mreza-tn (:normalized-matrix test-norm-input-310) @temp-variables3) 0)
    (restore-output-vector test-norm-target-310 (:normalized-matrix test-norm-target-310) 0)
    )
  )

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
(normalize-input-vector input-test2 norm-input-730 @norm-test)
(def norm-in (normalize-input-vector input-test norm-input-730 @norm-test))
(def norm-in2 (normalize-input-vector input-test2 norm-input-730 @norm-test))
(def norm-in3 (normalize-input-vector input-test3 norm-input-730 @norm-test))

(-> norm-in)

(def temp-variables4 (atom (create-temp-record @mreza-nn norm-in)))
(predict @mreza-nn norm-in @temp-variables4)

(restore-output-vector norm-target-730 (predict @mreza-nn norm-in2 @temp-variables4) 0)

(entry (restore-output-vector norm-target-730 (predict @mreza-nn norm-in3 @temp-variables4) 0) 0)

(predict @mreza-nn norm-in (create-temp-record @mreza-nn norm-in))

(predict @mreza-nn (:normalized-matrix norm-input-730) @temp-variables3)

(:normalized-matrix test-norm-input-310)

(:restore-coeficients norm-input-730)

;;
(def mreza-nn (atom (create-network-gaussian 64 [80 80 80] 1)))
(xavier-initialization-update @mreza-nn)
(set-biases-to-one @mreza-nn 0)

(def input-row-test (submatrix (:normalized-matrix input-trainig-dataset) 0 0 (inc (first (:config @mreza-nn))) 1))
(def temp-variables2 (atom (create-temp-record @mreza-nn input-row-test)))

(def temp-variables3 (atom (create-temp-record @mreza-nn (:normalized-matrix input-test-dataset))))

(time (train-network @mreza-nn (:normalized-matrix input-trainig-dataset) (:normalized-matrix target-trainig-dataset) 500
                     @temp-variables2 0.00137 0.9))


(evaluate-original-mape
  (evaluate-original
    (restore-output-vector target-test-dataset (predict @mreza-nn (:normalized-matrix input-test-dataset) @temp-variables3) 0)
    (restore-output-vector target-test-dataset (:normalized-matrix target-test-dataset) 0)
    ))

(predict @mreza-nn (:normalized-matrix input-test-dataset) (create-temp-record @mreza-nn (:normalized-matrix input-test-dataset)))