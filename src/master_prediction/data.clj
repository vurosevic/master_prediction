(ns ^{:author "Vladimir Urosevic"}
master-prediction.data
  (:require [clojure.string :as string]
            [uncomplicate.neanderthal.core :refer :all]
            [uncomplicate.neanderthal.vect-math :refer :all]
            [uncomplicate.neanderthal.native :refer :all]))

(defrecord Normalizedmatrix [
                             normalized-matrix
                             restore-coeficients            ;; matrix 2 x num of cols, min and max value for each columns
                             ])

(defn parse-float [s]
  (Float/parseFloat s))

(defn read-data-from-csv
  "Read the csv file, split out each line and then each number,
  parse the tokens and break up the numbers so that the last
  is the target and everything else is the feature vector."
  [filename]
  (as-> (slurp filename) d
        (string/split d #"\n")
        (map #(string/split % #",") d)
        (map #(map parse-float %) d)
        (map (fn [s] {:x (vec (drop-last s)) :y (vector (last s))}) d)))

(defn delete-file-if-exists
  [filename]
  (if (.exists (clojure.java.io/file (str "resources/" filename)))
    (clojure.java.io/delete-file (str "resources/" filename))))

(defn write-file [filename data]
  (with-open [w (clojure.java.io/writer (str "resources/" filename) :append true)]
    (.write w data)))

(defn save-network-to-file
  "save network state in file"
  [network filename]
  (do
    (delete-file-if-exists filename)
    (write-file filename "CONFIGURATION\n")
    (write-file filename
                (str (string/join ""
                                  (drop-last
                                    (reduce str (map str (:config network)
                                                     (replicate (count (:config network)) ","))))) "\n"))
    (write-file filename "LAYERS\n")
    (doall
      (doseq [y (range (count (:layers network)))]
        (write-file filename (str "LAYER," (inc y) "\n"))
        (doseq [x (range (ncols (nth (:layers network) y)))]
          (write-file filename
                      (str (string/join ""
                                        (drop-last
                                          (reduce str (map str (col (nth (:layers network) y) x)
                                                           (replicate (mrows (nth (:layers network) y)) ","))))) "\n")))))
    (write-file filename "END\n")))


(defn get-max-value
  "get max value from vector"
  [input-vec]
  (let [max-index (imax input-vec)]
    ;;(entry input-vec max-index)
    (* (entry input-vec max-index) 1.1)                     ;;1.1
    ))

(defn get-min-value
  "get min value from vector"
  [input-vec]
  (let [min-index (imin input-vec)]
    ;;(entry input-vec min-index)
    (* (entry input-vec min-index) 0.9)                     ;;0.9
    ))

(defn normalize-vector
  [input-vector input-min input-max result-vector]
  (let [temp-min-vec (fv (repeat (dim input-vector) input-min))
        temp-max-vec (fv (repeat (dim input-vector) input-max))
        temp-maxmin (fv (repeat (dim input-vector) input-max))
        temp-result (fv (repeat (dim result-vector) 0))
        temp-result2 (fv (repeat (dim result-vector) 0))]
    (do
      (axpy! input-vector temp-result)
      (axpy! -1 temp-min-vec temp-maxmin)
      (axpy! -1 temp-min-vec temp-result)
      (div! temp-result temp-maxmin temp-result2)
      (copy! temp-result2 result-vector)
      )))

(defn append-biases-to-normalized-matrix
  [norm-matrix]
  (let [dim-input-vector (mrows (:normalized-matrix norm-matrix))
        record-count (ncols (:normalized-matrix norm-matrix))
        temp-matrix-b (fge (inc dim-input-vector) record-count (repeat 1))
        - (copy! (:normalized-matrix norm-matrix) (submatrix temp-matrix-b 0 0 dim-input-vector record-count))]
    (->Normalizedmatrix temp-matrix-b
                        (:restore-coeficients norm-matrix))))

(defn create-norm-matrix
  [input-matrix]
  (let [rows-count (mrows input-matrix)
        cols-count (ncols input-matrix)
        norm-matrix (fge rows-count cols-count)
        coef-matrix (fge rows-count 2)]
    (do
      (doseq [x (range (mrows input-matrix))]
        (let [min-value (get-min-value (row input-matrix x))
              max-value (get-max-value (row input-matrix x))
              row-coef (row coef-matrix x)
              ]
          (do
            (entry! row-coef 0 min-value)
            (entry! row-coef 1 max-value)
            (normalize-vector (row input-matrix x)
                              min-value
                              max-value
                              (row norm-matrix x))
            )))
      (->Normalizedmatrix
        norm-matrix
        coef-matrix
        ))
    ))

(defn normalize-input-vector
  "normalize input vector"
  [input-vec norm-input-vector]
  (let [input (fge (count input-vec) 1 input-vec)
        coeficients (:restore-coeficients norm-input-vector)
        dim-coef (mrows (:restore-coeficients norm-input-vector))
        output-norm-vector (fge (inc (mrows input)) (ncols input) (repeat (* (inc (mrows input)) (ncols input)) 1))]
    (if (= (mrows input) dim-coef)
      (doseq [x (range dim-coef)]
        ;; (doseq [x (range (dec dim-coef))]
        (let [val-max (entry (row coeficients x) 1)
              val-min (entry (row coeficients x) 0)
              val-maxmin (- val-max val-min)
              val-input (entry (col input 0) x)
              val-norm (/ (- val-input val-min) val-maxmin)]
          (entry! (col output-norm-vector 0) x val-norm)
          )
        )
      (throw (Exception. (str "Error: Input vector does not match the norm input vector."))))
    (-> output-norm-vector)
    ))

(defn restore-output-vector
  "Restoring output vector"
  [normalized-record norm-matrix]
  (let [coef (:restore-coeficients normalized-record)
        row-no 0
        norm-vector (copy (row norm-matrix row-no))
        min-value (entry (row coef row-no) 0)
        max-value (entry (row coef row-no) 1)
        maxmin-value (- max-value min-value)
        maxmin-vector (fv (repeat (dim norm-vector) maxmin-value))
        min-vector (fv (repeat (dim norm-vector) min-value))]
    (axpy! min-vector (mul norm-vector maxmin-vector))
    ))

;; matrix with new data

(def data-file "resources/data_prediction.csv")
(def data-from-file (read-data-from-csv data-file))
(def dim-input-vector (count (first (map :x data-from-file))))
(def dim-output-vector (count (first (map :y data-from-file))))
(def record-count (count data-from-file))

(def input-matrix-all (fge dim-input-vector record-count (reduce into [] (map :x data-from-file))))
(def target-matrix-all (fge dim-output-vector record-count (reduce into [] (map :y data-from-file))))

;; prepare normalized data

(def norm-input-all (append-biases-to-normalized-matrix (create-norm-matrix input-matrix-all)))
(def norm-target-all (create-norm-matrix target-matrix-all))

(-> norm-input-all)
(-> norm-target-all)

(defn get-training-dataset
  "take training dataset by begining"
  [input-data percent]
  (let [max-num-rows (ncols (:normalized-matrix input-data))
        num-cols (mrows (:normalized-matrix input-data))
        num-rows (* (/ max-num-rows 100) percent)
        coeficients (:restore-coeficients input-data)
        norm-data (submatrix (:normalized-matrix input-data) 0 0 num-cols num-rows)
        ]
    (->Normalizedmatrix
      norm-data
      coeficients)))

(defn get-test-dataset
  "take test dataset from the end"
  [input-data percent]
  (let [max-num-rows (ncols (:normalized-matrix input-data))
        num-rows (* (/ max-num-rows 100) (- 100 percent))
        num-cols (mrows (:normalized-matrix input-data))
        coeficients (:restore-coeficients input-data)
        norm-data (submatrix (:normalized-matrix input-data) 0 num-rows num-cols (- max-num-rows num-rows))
        ]
    (->Normalizedmatrix
      norm-data
      coeficients)))

(def input-trainig-dataset (get-training-dataset norm-input-all 70))
(def input-test-dataset (get-test-dataset norm-input-all 30))

(def target-trainig-dataset (get-training-dataset norm-target-all 70))
(def target-test-dataset (get-test-dataset norm-target-all 30))