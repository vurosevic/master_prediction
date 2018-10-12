(ns ^{:author "Vladimir Urosevic"}
master-prediction.example
  (:require [uncomplicate.neanderthal.core :refer :all]
            [uncomplicate.neanderthal.vect-math :refer :all]
            [uncomplicate.neanderthal.native :refer :all]
            [uncomplicate.neanderthal.linalg :refer :all]
            [uncomplicate.commons.core :refer :all]
            [uncomplicate.clojurecl.constants :refer :all]
            [uncomplicate.neanderthal.internal.host.fluokitten :refer :all]
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

(gb native-float 80 80 (trans (first (:temp-vector-vector-h-gradients @temp-variables2))))

(mm
  (first (:temp-vector-vector-h-gradients @temp-variables2))
  (gb native-float 80 80 (first (:temp-vector-vector-h-gradients @temp-variables2)))
  )


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

(time (train-network @mreza-nn (:normalized-matrix norm-input-730) (:normalized-matrix norm-target-730) 200
                     @temp-variables2 0.0015 0.9))

(evaluate-original-mape
  (evaluate-original
    (restore-output-vector test-norm-target-310 (predict @mreza-nn (:normalized-matrix test-norm-input-310) @temp-variables3) 0)
    (restore-output-vector test-norm-target-310 (:normalized-matrix test-norm-target-310) 0)
    )
  )

(restore-output-vector test-norm-target-310 (predict @mreza-nn (:normalized-matrix test-norm-input-310) @temp-variables3) 0)

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
