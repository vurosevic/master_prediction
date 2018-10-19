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
                                    (reduce str (map str (:config network)
                                            (replicate (count (:config network)) ",")) ))) "\n"))


    (write-file filename "LAYERS\n")
    (doall

      (doseq [y (range (count (:layers network)))]
        (write-file filename (str "LAYER," (inc y) "\n"))
        (doseq [x (range (mrows (nth (:layers network) y)))]
          (write-file filename
                      (str (string/join ""
                                        (drop-last
                                          (reduce str (map str (row (nth (:layers network) y) x)
                                                      (replicate (ncols (nth (:layers network) y)) ","))))) "\n"))))
      )

    (write-file filename "END\n")))


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

  (let [temp-min-vec (fv (repeat (dim input-vector) input-min)) ;; (/ input-min 0.9)
        temp-max-vec (fv (repeat (dim input-vector) input-max)) ;; (/ input-max 1.1)
        temp-maxmin  (fv (repeat (dim input-vector) input-max)) ;; (/ input-max 1.1)
        temp-result  (fv (repeat (dim result-vector) 0))
        temp-result2  (fv (repeat (dim result-vector) 0))
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
        input-matrix-b (submatrix input-matrix 0 0 (dec rows-count) cols-count)

        norm-matrix (fge rows-count cols-count (repeat 1))             ;; create null matrix
        norm-matrix-b (submatrix norm-matrix 0 0 (dec rows-count) cols-count) ;; create null matrix
        coef-matrix (fge rows-count 2)
        ]

    (do
      (doseq [x (range (mrows input-matrix-b))]
        (let [min-value (get-min-value (row input-matrix-b x))
              max-value (get-max-value (row input-matrix-b x))
              row-coef  (row coef-matrix x)
              ]
          (do
            (entry! row-coef 0 min-value)
            (entry! row-coef 1 max-value)
            (normalize-vector (row input-matrix-b x)
                              min-value
                              max-value
                              (row norm-matrix-b x))
            )
          )
        )

      (->Normalizedmatrix
        norm-matrix
        coef-matrix
        ))
    )
  )

(defn create-norm-matrix-target
  [input-matrix]
  (let [rows-count (mrows input-matrix)
        cols-count (ncols input-matrix)
        norm-matrix (fge rows-count cols-count)             ;; create null matrix
        coef-matrix (fge rows-count 2)]
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

(defn normalize-input-vector
  "normalize input vector"
  [input-vec norm-input-vector output-norm-vector]
  (let [input (fge (count input-vec) 1 input-vec)
        coeficients (:restore-coeficients norm-input-vector)
        dim-coef (mrows (:restore-coeficients norm-input-vector))
        output-norm-vector (fge (mrows input) (ncols input) (repeat (* (mrows input) (ncols input)) 1))]
    (if (= (mrows input) dim-coef)
      (doseq [x (range (dec dim-coef))]
        (let [val-max (entry (row coeficients x) 1)
              val-min (entry (row coeficients x) 0)
              val-maxmin (- val-max val-min)
              val-input (entry (col input 0) x)
              val-norm  (/ (- val-input val-min) val-maxmin)]
          (entry! (col output-norm-vector 0) x val-norm)
         )
       )
      (throw (Exception. (str "Error: Input vector does not match the norm input vector."))))

      (-> output-norm-vector)
    )
  )

(defn restore-output-vector
  "Restoring output vector"
  [normalized-record norm-matrix row-no]
  (let [coef (:restore-coeficients normalized-record)
        norm-vector (copy (row norm-matrix row-no))
        min-value (entry (row coef row-no) 0)
        max-value (entry (row coef row-no) 1)
        maxmin-value (- max-value min-value)
        maxmin-vector (fv (repeat (dim norm-vector) maxmin-value))
        min-vector (fv (repeat (dim norm-vector) min-value))]

    (axpy! min-vector (mul norm-vector maxmin-vector))
    )
  )

;; matrix with new data
(def input-matrix-all (fge 64 2647 (reduce into [] (map :x (read-data-from-csv "resources/data_prediction.csv")))))
(def target-matrix-all (fge 1 2647 (reduce conj [] (map :y (read-data-from-csv "resources/data_prediction.csv")))))

(def input-730 (submatrix input-matrix-all 0 0 64 1852))
(def target-730 (submatrix target-matrix-all 0 0 1 1852))

;;append 1 row for biases inputs
(def input-730-b (fge 65 1852 (repeat 1)))
(copy! input-730 (submatrix input-730-b 0 0 64 1852))


(def test-input-310 (submatrix input-matrix-all 0 1854 64 793))
(def test-target-310 (submatrix target-matrix-all 0 1854 1 793))

;;append 1 row for biases inputs
(def test-input-310-b (fge 65 793 (repeat 1)))
(copy! test-input-310 (submatrix test-input-310-b 0 0 64 793))


;; normalized data
;; training
(def norm-input-730 (create-norm-matrix input-730-b))
(def norm-target-730 (create-norm-matrix-target target-730))

;; test
(def test-norm-input-310 (create-norm-matrix test-input-310-b))
(def test-norm-target-310 (create-norm-matrix-target test-target-310))

;; prepare normalized data

;; normalizovati sve podatke, pa zatim uzeti deo za obuku 70% i deo za test 30%
;;


(def input-all-b (fge 65 2647 (repeat 1)))
(copy! input-matrix-all (submatrix input-all-b 0 0 64 2647))

(def norm-input-all (create-norm-matrix input-all-b))
(def norm-target-all (create-norm-matrix-target target-matrix-all))

(ncols (:normalized-matrix norm-input-all))

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

(-> norm-input-all)
(-> norm-target-all)

(def input-trainig-dataset (get-training-dataset norm-input-all 70))
(def input-test-dataset (get-test-dataset norm-input-all 30))

(def target-trainig-dataset (get-training-dataset norm-target-all 70))
(def target-test-dataset (get-test-dataset norm-target-all 30))