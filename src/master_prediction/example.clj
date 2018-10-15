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

(vec (map #(col (get-bias-vector %) 0) (:layers @mreza-nn)))
(col (nth (vec (map #(get-bias-vector %) (:layers @mreza-nn))) 0) 0)

(feed-forward @mreza-nn (:normalized-matrix norm-input-730) @temp-variables2)

(predict @mreza-nn (:normalized-matrix norm-input-730) @temp-variables2)

(restore-output-vector norm-target-730 (predict @mreza-nn (:normalized-matrix norm-input-730) @temp-variables2) 0)

(def temp-variables3 (atom (create-temp-record @mreza-nn (:normalized-matrix test-norm-input-310))))
(predict @mreza-nn (:normalized-matrix test-norm-input-310) @temp-variables3)


;; prebacivanje vektora u diag matricu

(def tt-v (gb native-float 3 3 (dv [1 2 3])))
(def tt-m (fge 3 1 [1 2 3]))

(-> tt-v)

(mm tt-v tt-m)

(quick-bench (mm tt-v tt-m))

(gb native-float 3 3 (dv [1 2 3]))
(fge 3 1 [1 2 3])

(quick-bench (fge 3 1 [1 2 3]))
(quick-bench (dge 3 1 [1 2 3]))

(quick-bench (gb native-float 3 3 [1 2 3]))

(gb native-float (first (:temp-vector-vector-h-gradients @temp-variables2)))


(first (:temp-vector-vector-h-gradients @temp-variables2))
(trans (first (:temp-vector-vector-h-gradients @temp-variables2)))

;; reseno !!!
(mm
  (gb native-float (first (:temp-vector-vector-h-gradients @temp-variables2)))
  (trans (first (:temp-vector-vector-h-gradients @temp-variables2)))
  )

(:temp-vector-vector-h-gradients @temp-variables3)

(quick-bench
  (mul (:temp-vector-o-gradients @temp-variables2) (:temp-vector-o-gradients2 @temp-variables2))
  )

(mrows (:normalized-matrix norm-input-730))

(for [x (range 1000)]
  (backpropagation @mreza-nn (:normalized-matrix norm-input-730) 0 (:normalized-matrix norm-target-730)
                   @temp-variables2 0.015 0)
  )
(nth (:layers @mreza-nn) 0)
(mrows (nth (:layers @mreza-nn) 0))                         ;;81
(ncols (nth (:layers @mreza-nn) 0))                         ;;65
(submatrix (nth (:layers @mreza-nn) 0) 0 64 80 1)
(get-bias-vector (nth (:layers @mreza-nn) 0))

(nth (:temp-vector-matrix-delta @temp-variables2) 0)

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

(count (:config @mreza-nn))

(time (train-network @mreza-nn (:normalized-matrix norm-input-730) (:normalized-matrix norm-target-730) 100
                     @temp-variables2 0.0015 0.9))

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

(:normalized-matrix test-norm-target-310)

(submatrix (:normalized-matrix norm-input-730) 0 0 (inc (first (:config @mreza-nn))) 1)
(def input-row-test (submatrix (:normalized-matrix norm-input-730) 0 0 (inc (first (:config @mreza-nn))) 1))
(def temp-variables2 (atom (create-temp-record @mreza-nn input-row-test)))
(feed-forward @mreza-nn input-row-test @temp-variables2)

(-> @temp-variables2)
(last (:layers-output @temp-variables2))
(nth (:layers @mreza-nn) 3)
(get-weights-matrix (nth (:layers @mreza-nn) 2))
(nth (:temp-vector-vector-h-gradients @temp-variables2) 2)
(mm (trans (get-weights-matrix (nth (:layers @mreza-nn) 2))) (nth (:temp-vector-vector-h-gradients @temp-variables2) 2))
(nth (:temp-vector-vector-h-gradients @temp-variables2) 2)
(get-output-matrix (last (:layers-output @temp-variables2)))
(col (nth (:temp-vector-matrix-delta @temp-variables2) 0) 0)
(col (nth (conj (:layers-output @temp-variables2)  (submatrix input-row-test 0 0 64 1)) 0) 0)

(:normalized-matrix norm-input-730)
(:normalized-matrix norm-target-730)

(def tt (predict @mreza-nn (:normalized-matrix test-norm-input-310) @temp-variables3))


(nth (:layers-output @temp-variables3) 3)
(:restore-coeficients test-norm-input-310)
(:normalized-matrix test-norm-input-310)

(restore-output-vector test-norm-target-310 (predict @mreza-nn (:normalized-matrix test-norm-input-310) @temp-variables3) 0)


(submatrix (nth (:layers-output @temp-variables2) 3) 0 0 1 1852)

(first (:config @mreza-nn))
(mrows (:normalized-matrix norm-input-730))

(with-progress-reporting
  (quick-bench (feed-forward @mreza-nn (:normalized-matrix norm-input-730) @temp-variables2))
  )



(mrows input-730-b)
(first (:config @mreza-nn))
(last (:config @mreza-nn))


(layer-output (:normalized-matrix norm-input-730) (nth (:layers @mreza-nn) 0)
              (nth (:layers-output @temp-variables2) 0) tanh!)

(:normalized-matrix norm-input-730)
(:layers-output @temp-variables2)

(nth (:layers @mreza-nn) 0)
(trans (nth (:layers @mreza-nn) 0))

(trans (:normalized-matrix norm-input-730))

(nth (:layers-output @temp-variables2) 3)

(:layers @mreza-nn)
(:layers-output @temp-variables2)

(ncols input-730-b)

(nth (:layers-output @mreza-nn) 3)

;; proba
(def mm-proba (fge 3 3 [1 1 1 2 2 2 3 3 3]))
(def mm-proba-trans (trans mm-proba))
(-> mm-proba)
(def rm-proba (row mm-proba 1))
(-> rm-proba)
(entry! (row mm-proba 1) 2 9)
(-> mm-proba-trans)

(-> @mreza-nn)

(-> temp-variables3)
(vec (map #(get-output-matrix %) (:layers-output @temp-variables3)))

(vec (map #(get-weights-matrix %) (:layers @mreza-nn)))

(vec (map #(trans (get-weights-matrix %)) (:layers @mreza-nn)))

(map #(ncols %) (:temp-vector-matrix-delta @temp-variables3))

(map #(mrows %) (:temp-vector-vector-h-gradients @temp-variables3))

(take-last (dec (count (:config @mreza-nn))) (:config @mreza-nn))

(save-network-to-file @mreza-nn "test-mreza2.csv")

(load-network-config "test-mreza3.csv")

(load-network-layers "test-mreza3.csv" 1)

(def mreza-tn (atom (create-network-from-file "test-mreza2.csv")))
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

(str (string/join ""
                  (drop-last
                    (reduce #(str %1 %2)
                            (map str (:config @mreza-nn)
                                 (replicate (count (:config @mreza-nn)) ","))
                            )
                  )
      )
"\n")

(map inc [1 2 3])