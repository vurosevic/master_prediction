(ns ^{:author "Vladimir Urosevic"}
master-prediction.data
  (:require [clojure.string :as string]
            [uncomplicate.neanderthal.core :refer :all]
            [uncomplicate.neanderthal.vect-math :refer :all]
            [uncomplicate.neanderthal.native :refer :all]))

(defrecord Normalizedmatrix [
                             normalized-matrix
                             restore-coeficients                ;; matrix 2 x num of cols, min and max value for each columns
                             ]
  )

(defn parse-float [s]
  (Float/parseFloat s)
  )

(defn read-data-from-csv
  "Read the csv file, split out each line and then each number, parse the tokens and break up the numbers so that the last is the target and everything else is the feature vector."
  [filename]
  (as-> (slurp filename) d
        (string/split d #"\n")
        (map #(string/split % #",") d)
        (map #(map parse-float %) d)
        (map (fn [s] {:x  (vec (drop-last s))  :y (last s)}) d)))

(defn write-file [filename data]
  (with-open [w (clojure.java.io/writer  (str "resources/" filename) :append true)]
    (.write w data)))

(defn save-network-to-file
  "save network state in file"
  [network filename]
  (do
    (write-file filename "CONFIGURATION\n")
    (write-file filename
                (str (string/join ""
                                  (drop-last
                                    (reduce str (map str (conj (vec (:tmp1 network)) (last (:tmp2 network)))
                                                     (replicate (count (conj (vec (:tmp1 network)) (last (:tmp2 network)))) ","))))) "\n")
                )

    (write-file filename "BIASES\n")
    (doall

      (doseq [y (range (count (:biases network)))]
        (write-file filename (str "BIAS," (inc y) "\n"))
        (doseq [x (range (mrows (nth (:biases network) y)))]
          (write-file filename
                      (str (string/join ""
                                        (drop-last
                                          (reduce str (map str (row (nth (:biases network) y) x)
                                                           (replicate (ncols (nth (:biases network) y)) ","))))) "\n"))))
      )

    (write-file filename "LAYERS\n")
    (doall

      (doseq [y (range (count (:hidden-layers network)))]
        (write-file filename (str "LAYER," (inc y) "\n"))
        (doseq [x (range (mrows (nth (:hidden-layers network) y)))]
          (write-file filename
                      (str (string/join ""
                                        (drop-last
                                          (reduce str (map str (row (nth (:hidden-layers network) y) x)
                                                           (replicate (ncols (nth (:hidden-layers network) y)) ","))))) "\n"))))
      )

    (write-file filename "OUTPUT\n")
    (doall
      (for [x (range (mrows (:output-layer network)))]
        (write-file filename
                    (str (string/join ""
                                      (drop-last
                                        (reduce str (map str (row (:output-layer network) x)
                                                         (replicate (ncols (:output-layer network)) ","))))) "\n"))))
    (write-file filename "END\n")
    ))


(defn get-max-value
  "get max value from vector"
  [input-vec]
  (let [max-index (imax input-vec)]
    (* (entry input-vec max-index) 1.1)
    ))

(defn get-min-value
  "get min value from vector"
  [input-vec]
  (let [min-index (imin input-vec)]
    (* (entry input-vec min-index) 0.9)
    ))

(defn normalize-vector
  [input-vector input-min input-max result-vector]

  (let [temp-min-vec (dv (repeat (dim input-vector) input-min)) ;; (/ input-min 0.9)
        temp-max-vec (dv (repeat (dim input-vector) input-max)) ;; (/ input-max 1.1)
        temp-maxmin  (dv (repeat (dim input-vector) input-max)) ;; (/ input-max 1.1)
        temp-result  (dv (repeat (dim result-vector) 0))
        temp-result2  (dv (repeat (dim result-vector) 0))
        ]

    (do
      (axpy! input-vector temp-result)
      (axpy! -1 temp-min-vec temp-maxmin)
      (axpy! -1 temp-min-vec temp-result)
      (div! temp-result temp-maxmin temp-result2)
      (copy! temp-result2 result-vector)
      )
    )
  )

(defn create-norm-matrix
  [input-matrix]
  (let [rows-count (mrows input-matrix)
        cols-count (ncols input-matrix)
        norm-matrix (dge rows-count cols-count)             ;; create null matrix
        coef-matrix (dge rows-count 2)
        ]

    (do
      (doseq [x (range (mrows input-matrix))]
        (let [min-value (get-min-value (row input-matrix x))
              max-value (get-max-value (row input-matrix x))
              row-coef  (row coef-matrix x)
              ]
          (do
            (entry! row-coef 0 min-value)
            (entry! row-coef 1 max-value)
            (normalize-vector (row input-matrix x)
                              min-value
                              max-value
                              (row norm-matrix x))
            )
          )
        )

      (->Normalizedmatrix
        norm-matrix
        coef-matrix
        ))
    )
  )

(defn restore-output-vector
  "Restoring output vector"
  [normalized-record norm-matrix row-no]
  (let [coef (:restore-coeficients normalized-record)
        norm-vector (row norm-matrix row-no)
        min-value (entry (row coef row-no) 0)
        max-value (entry (row coef row-no) 1)
        maxmin-value (- max-value min-value)
        maxmin-vector (dv (repeat (dim norm-vector) maxmin-value))
        min-vector (dv (repeat (dim norm-vector) min-value))
        ]

    (axpy! min-vector (mul norm-vector maxmin-vector))
    )
  )

;; matrix with new data
(def input-matrix-all (dge 64 2647 (reduce into [] (map :x (read-data-from-csv "resources/data_prediction.csv")))))
(def target-matrix-all (dge 1 2647 (reduce conj [] (map :y (read-data-from-csv "resources/data_prediction.csv")))))

(def input-730 (submatrix input-matrix-all 0 0 64 1852))
(def target-730 (submatrix target-matrix-all 0 0 1 1852))

(def test-input-310 (submatrix input-matrix-all 0 1854 64 793))
(def test-target-310 (submatrix target-matrix-all 0 1854 1 793))

(def norm-input-730 (create-norm-matrix input-730))
(def norm-target-730 (create-norm-matrix target-730))

(def test-norm-input-310 (create-norm-matrix test-input-310))
(def test-norm-target-310 (create-norm-matrix test-target-310))

(native! norm-input-730)
(native! norm-target-730)
(native! test-norm-input-310)
(native! test-norm-target-310)


