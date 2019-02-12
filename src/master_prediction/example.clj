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

;; snimanje i ucitavanje konfiguracije
(def mreza-nn (atom (create-network-from-file "nn-mreza-122.csv")))
(save-network-to-file @mreza-nn "nn-mreza-126.csv")

(-> @mreza-nt)

;; predikcija za specificirani ulazni vektor
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


(quick-bench
  (predict @mreza-nn norm-in @temp-variables)
  )

(with-progress-reporting
  (quick-bench
    (restore-output-vector target-trainig-dataset (predict @mreza-nn norm-in @temp-variables))
    )
  )


;; kreiranje i treniranje mreze
(def mreza-nn (atom (create-network 64 [100 200 100] 1)))
(def mreza-nn (atom (create-network 64 [100 200 200 100] 1)))
(def mreza-nn (atom (create-network 64 [200 200] 1)))
(def mreza-nn (atom (create-network 64 [100 100] 1)))
(def mreza-nn (atom (create-network 64 [100] 1)))
;;(def mreza-nn (atom (create-network 64 [100 100 100 100] 1)))
;; (xavier-initialization-update @mreza-nn)
;; (set-biases-value @mreza-nn 0)

(def temp-variables3 (atom (create-temp-record @mreza-nn (:normalized-matrix input-test-dataset))))

(quick-bench (train-network @mreza-nn (:normalized-matrix input-trainig-dataset)
                     (:normalized-matrix target-trainig-dataset) 1 2
                     0.01708557 0.9))

(quick-bench (train-network @mreza-nn (:normalized-matrix input-trainig-dataset)
                            (:normalized-matrix target-trainig-dataset) 1 1
                            0.0015557 0.9))

(time (train-network @mreza-nn (:normalized-matrix input-trainig-dataset)
                     (:normalized-matrix target-trainig-dataset) 1 1
                     0.0015557 0.9))

(time (train-network @mreza-nn (:normalized-matrix input-trainig-dataset)
                               (:normalized-matrix target-trainig-dataset) 20 1
                               0.0015557 0.9))

(time (train-network @mreza-nn (:normalized-matrix input-trainig-dataset)
                     (:normalized-matrix target-trainig-dataset) 200 2
                     0.0170807557 0.9))

(time (train-network @mreza-nn (:normalized-matrix input-trainig-dataset)
                     (:normalized-matrix target-trainig-dataset) 20 15
                     0.0001570807557 0.9))

;; test
(:normalized-matrix input-trainig-dataset)
(:normalized-matrix target-trainig-dataset)
(def temp-vars (create-temp-record @mreza-nn (:normalized-matrix input-trainig-dataset)))
(def temp-vars2 (create-temp-record @mreza-nn (:normalized-matrix input-test-dataset)))
(:layers-output-only temp-vars)
(last (:layers-output-only temp-vars))
(:layers-output-only temp-vars2)
(axpy -1 (:normalized-matrix target-trainig-dataset) (last (:layers-output-only temp-vars)))

(dtanh! (:normalized-matrix target-trainig-dataset) (last (:layers-output-only temp-vars)))

(last (:temp-all-vector-vector-h-gradients temp-vars))

(last (:temp-all-vector-vector-h-signals temp-vars))



(nth (:temp-all-vector-vector-h-signals temp-vars) 2)
 (get-weights-matrix (nth (:layers @mreza-nn) 3))

(get-weights-matrix (nth (:layers @mreza-nn) 3))

(sum (get-weights-matrix (nth (:layers @mreza-nn) 2)))

(mm (trans (nth (:temp-all-vector-vector-h-signals temp-vars) 3))
    (get-weights-matrix (nth (:layers @mreza-nn) 3))
    )

(mm (trans (nth (:temp-all-vector-vector-h-signals temp-vars) 0))
    (get-weights-matrix (nth (:layers @mreza-nn) 0))
    )

(ncols (mm (trans (nth (:temp-all-vector-vector-h-signals temp-vars) 3))
           (get-weights-matrix (nth (:layers @mreza-nn) 3))
           ))

(dim (col (last (:temp-all-vector-vector-h-signals temp-vars)) 1))

(def temp-matrix-only (:layers-output-only temp-vars))
(nth temp-matrix-only 2)

(mul (nth temp-matrix-only 2) (nth temp-matrix-only 2))

(ncols (get-weights-matrix (nth (:layers @mreza-nn) 3)))
(range (- (count (:layers @mreza-nn)) 1) 0 -1)

;; kreiranje dijagonalne matrice
(fgd 1852 (last (:temp-all-vector-vector-h-signals temp-vars)) {:row :column})

(row (fgd 1852 (last (:temp-all-vector-vector-h-signals temp-vars)) {:column :row}) 1)

(mm
  (fgd 1852 (last (:temp-all-vector-vector-h-signals temp-vars)))
  (nth temp-matrix-only 2)
  )

(:temp-vector-matrix-delta temp-vars)

(col (mm (nth temp-matrix-only 2)

         ) 0)

(mm (submatrix unit-matrix 0 0 100 1)
    (nth (:temp-all-vector-vector-h-signals temp-vars) 3))

(nth (:temp-all-vector-vector-h-signals temp-vars) 2)
(:temp-all-vector-vector-h-gradients temp-vars)
(:temp-all-vector-vector-h-signals temp-vars)

(:temp-prev-delta-vector-matrix-delta temp-vars)

(:layers-output-only temp-vars)

(submatrix unit-matrix 0 0 100 1)
(nth (:temp-all-vector-vector-h-signals temp-vars) 3)
(nth temp-matrix-only 1)

(prepare-identity-matrix 1)
(nth (:temp-all-vector-vector-h-signals temp-vars) 3)
( ncols (nth (:temp-all-vector-vector-h-signals temp-vars) 3))

(mm
  (nth (:temp-all-vector-vector-h-signals temp-vars) 3)
  (submatrix unit-matrix 0 0 1852 1)
  )

  (prepare-identity-matrix 1852)

(def pr (copy (prepare-identity-matrix 1852)))
(-> pr)
(axpy! (row (nth (:temp-all-vector-vector-h-signals temp-vars) 3) 0)
       (dia pr))

( (nth (:temp-all-vector-vector-h-signals temp-vars) 3) (dia (prepare-identity-matrix 1852)) )


(nth (:temp-all-vector-vector-h-signals temp-vars) 2)
(nth temp-matrix-only 1)

(nth temp-matrix-only 2)

(mm
  (nth (:temp-all-vector-vector-h-signals temp-vars) 2)
  (submatrix unit-matrix 0 0 1852 1)
  )

(quick-bench
  (copy (prepare-zero-matrix 100))
  )

(def ttt (prepare-zero-matrix 100))

(quick-bench
  (entry! (dia ttt) 0)
  )

(:temp-prev-delta-vector-matrix-delta temp-vars)
(:temp-all-vector-vector-h-signals temp-vars)

(:temp-vector-matrix-delta-biases temp-vars)

(nth (:temp-all-vector-vector-h-signals temp-vars) 2)

(entry (row (last (:layers-output temp-vars)) 0) 0)
(entry (row (last (:layers-output temp-vars)) 0) 1)
(def pp (entry (row (last (:layers-output temp-vars)) 0) 1))

(* (+ 1 pp) (- 1 pp))



(sum (last (:temp-all-vector-vector-h-signals temp-vars)))
(entry (row (nth (:layers @mreza-nn) 2) 0) 0)
(entry (row (nth (:layers @mreza-nn) 2) 0) 1)

(:temp-all-vector-vector-h-signals temp-vars)
(:temp-all-vector-vector-h-gradients temp-vars2)
(:temp-vector-matrix-delta-biases temp-vars)
(:temp-vector-matrix-delta temp-vars)
(:temp-vector-matrix-delta temp-vars2)
(:temp-prev-delta-vector-matrix-delta temp-vars)

(:temp-vector-vector-h-signals-var1 temp-vars)
(:temp-vector-vector-h-signals-var2 temp-vars)

(:normalized-matrix input-trainig-dataset)


(def mreza-nn2 (atom (create-network 64 [100 200 100] 1)))

(def mreza-nn (atom (create-network 64 [100 200 200 100] 1)))
(def mreza-nn (atom (create-network 64 [100 200 100] 1)))
(def mreza-nn (atom (create-network 64 [200 200] 1)))
(def mreza-nn (atom (create-network 64 [100 100] 1)))
(def mreza-nn (atom (create-network 64 [100] 1)))

(:layers @mreza-nn)0

(time (train-network @mreza-nn (:normalized-matrix input-trainig-dataset)
                     (:normalized-matrix target-trainig-dataset) 300 20
                     0.0157 0.9))

(with-progress-reporting
  (quick-bench (train-network @mreza-nn (:normalized-matrix input-trainig-dataset)
                              (:normalized-matrix target-trainig-dataset) 1 5
                              0.01708557 0.9))
  )


(:layers-output temp-vars)
(def temp-vars (create-temp-record @mreza-nn (:normalized-matrix input-trainig-dataset)))
(def temp-vars2 (create-temp-record @mreza-nn2 (:normalized-matrix input-trainig-dataset)))
(learning-once2 @mreza-nn (:normalized-matrix input-trainig-dataset) 1
                (:normalized-matrix target-trainig-dataset)
                temp-vars 0.0015 0 1)

(learning-once @mreza-nn2 (:normalized-matrix input-trainig-dataset) 1
                (:normalized-matrix target-trainig-dataset)
                temp-vars2 0.0015 0)

(quick-bench
  (learning-once2 @mreza-nn (:normalized-matrix input-trainig-dataset) 1
                  (:normalized-matrix target-trainig-dataset)
                  temp-vars 0.001 0.9 10)
  )

(fgd 5 [1 1 1 1])
;; end test



(save-network-to-file @mreza-nn "test-1-01.csv")


(def mreza-nn (atom (create-network-from-file "early-stopping-net.csv")))

(let [pred-values (restore-output-vector target-test-dataset (predict @mreza-nn (:normalized-matrix input-test-dataset) @temp-variables3))
      values (restore-output-vector target-test-dataset (:normalized-matrix target-test-dataset))
      count-values (dim pred-values)]
  (doseq [x (range count-values)]
    (write-file "prognoza_1_01.csv" (str x "," (entry values x) "," (entry pred-values x) "\n"))
    )
  )

(restore-output-vector target-test-dataset (predict @mreza-nn (:normalized-matrix input-test-dataset) @temp-variables3))
(restore-output-vector target-test-dataset (:normalized-matrix target-test-dataset))

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

;; evaluacija mreze, test podaci
(evaluate-mape
  (evaluate
    (restore-output-vector target-test-dataset (predict @mreza-nn (:normalized-matrix input-test-dataset) @temp-variables3))
    (restore-output-vector target-test-dataset (:normalized-matrix target-test-dataset))
    ))

(evaluate
  (restore-output-vector target-test-dataset (predict @mreza-nn (:normalized-matrix input-test-dataset) @temp-variables3))
  (restore-output-vector target-test-dataset (:normalized-matrix target-test-dataset))
  )

(predict @mreza-nn (:normalized-matrix input-test-dataset) @temp-variables3)
(restore-output-vector target-test-dataset (predict @mreza-nn (:normalized-matrix input-test-dataset) @temp-variables3))

;; evaluacija sa trening podacima
(def temp-variables5 (atom (create-temp-record @mreza-nn (:normalized-matrix input-trainig-dataset))))
(evaluate-mape
  (evaluate
    (restore-output-vector target-trainig-dataset (predict @mreza-nn (:normalized-matrix input-trainig-dataset) @temp-variables5))
    (restore-output-vector target-trainig-dataset (:normalized-matrix target-trainig-dataset))
    ))


(evaluate
  (restore-output-vector target-test-dataset (predict @mreza-nn (:normalized-matrix input-test-dataset) @temp-variables3))
  (restore-output-vector target-test-dataset (:normalized-matrix target-test-dataset))
  )

(restore-output-vector target-test-dataset (predict @mreza-nn (:normalized-matrix input-test-dataset) @temp-variables3))
(restore-output-vector target-test-dataset (:normalized-matrix target-test-dataset))

(def nn (evaluate
          (restore-output-vector target-test-dataset (predict @mreza-nn (:normalized-matrix input-test-dataset) @temp-variables3))
          (restore-output-vector target-test-dataset (:normalized-matrix target-test-dataset))
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