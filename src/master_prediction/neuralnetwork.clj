(ns ^{:author "Vladimir Urosevic"}
master-prediction.neuralnetwork
  (:require [master-prediction.data :refer :all]
            [uncomplicate.neanderthal.core :refer :all]
            [uncomplicate.neanderthal.vect-math :refer :all]
            [uncomplicate.neanderthal.native :refer :all]
            [clojure.string :as string]
            [clojure.core :as core]))

(defrecord Tempvariable [
                         layers-output              ;; output for layers + 1 for biases for next leyer
                         layers-output-only         ;; output for layers

                         trans-weights              ;; trans weights matrixes

                         temp-all-vector-o-gradients             ;; matrix, row=1, output gradient, dim number of output neurons
                         temp-all-vector-o-gradients2            ;; matrix, row=1, output gradient, dim number of output neurons
                         temp-all-vector-vector-h-gradients      ;; output gradient, dim number of output neurons

                         temp-vector-o-gradients             ;; matrix, row=1, output gradient, dim number of output neurons
                         temp-vector-o-gradients2            ;; matrix, row=1, output gradient, dim number of output neurons
                         temp-vector-vector-h-gradients      ;; output gradient, dim number of output neurons

                         temp-vector-matrix-delta            ;; delta weights for layers
                         temp-vector-matrix-delta-biases     ;; delta biases for layers
                         temp-prev-delta-vector-matrix-delta ;; previous delta vector matrix layers
                         temp-vector-matrix-delta-momentum   ;; delta weights for layers - momentum
                         ])

(defrecord Neuronetwork [
                         layers                              ;; vector of layers (hiddens and output weights and biases)
                         config                              ;; vector, numbers of neurons by layer
                         layers-output                       ;; vector of matrix, outputs from layers
                         ])

(def max-dim 4096)

(def unit-vector (dv (replicate max-dim 1)))
(def unit-matrix (fge max-dim max-dim (repeat 1)))

(defn prepare-unit-vector
  "preparing unit vector for other calculations"
  [n]
  (if (<= n max-dim)
    (subvector unit-vector 0 n)
    (dv [0])))

(import '(java.util Random))
(def normals
  (let [r (Random.)]
    (map #(/ % 4.7) (take 500000 (repeatedly #(-> r .nextGaussian (* 0.9) (+ 1.0)))))
    ))


(def temp-current-val (atom 0))
(defn create-random-matrix-by-gaussian
  "Initialize a layer by gaussian"
  [dim-y dim-x]
  (do
    (if (> dim-y max-dim)
      (throw (Exception. (str "Error. Max number of neurons is " max-dim))))
    (if (> dim-x max-dim)
      (throw (Exception. (str "Error. Max number of neurons is " max-dim))))
    (reset! temp-current-val (+ @temp-current-val (* dim-x dim-y)))
    (fge dim-y dim-x (nthrest normals @temp-current-val))
    ;;(trans (fge dim-x dim-y (nthrest normals @temp-current-val)))
    ))

(defn create-null-matrix
  "Initialize a layer"
  [dim-y dim-x]
  (do
    (if (> dim-y max-dim)
      (throw (Exception. (str "Error. Max number of neurons is " max-dim))))
    (if (> dim-x max-dim)
      (throw (Exception. (str "Error. Max number of neurons is " max-dim))))
    (fge dim-y dim-x))
  )

(defn layer-output
  "generate output from layer"
  [input weights result o-func]
    (mm! 1.0 weights input 0.0 result)
    (o-func (submatrix result 0 0 (dec (mrows result)) (ncols result)))
  )

(defn dtanh!
  "calculate dtanh for vector or matrix"
  [y result]
  (if (matrix? y)
    (let [unit-mat (submatrix unit-matrix (mrows y) (ncols y))]
      (do (sqr! y result)
          (axpy! -1 unit-mat result)
          (scal! -1 result)))

    (let [unit-vec (subvector unit-vector 0 (dim y))]
      (do (sqr! y result)
          (axpy! -1 unit-vec result)
          (scal! -1 result)))
    )
  )

(defn prepare-last-col
  "Sets last column to [0 0 0 ... 1]"
  [input-matrix]
  (let [last-col (mrows input-matrix)
        col (row input-matrix (dec last-col))]
    (do
      (scal! 0 col)
      (entry! col (dec (dim col)) 1)
      )
    )
  )

(defn get-weights-matrix
  "return weights matrix from layer"
  [layer]
  (let [rows-num (mrows layer)
        cols-num (ncols layer)]
    (-> (submatrix layer 0 0 (dec rows-num) (dec cols-num)))
    ))

(defn get-output-matrix
  "function cut 1 from the end"
  [matrix]
  (let [rows-num (mrows matrix)
        cols-num (ncols matrix)]
    (submatrix matrix 0 0 (dec rows-num) cols-num)
    )
  )

(defn get-bias-vector
  "return biases from layer"
  [layer]
  (let [num-rows (dec (mrows layer))
        num-cols (dec (ncols  layer))]
    (submatrix layer 0 num-cols num-rows 1))
  )

(defn set-biases-to-one
  ""
  [network value]
  (let [layers (:layers network)
        num-layers (count layers)
        biases (vec (map #(col (get-bias-vector %) 0) layers))]
    (doseq [x (range (count biases))]
            (entry! (nth biases x) value)
        )
      )
    )


(defn create-network-gaussian
  "create new neural network"
  [number-input-neurons vector-of-numbers-hidden-neurons number-output-neurons]
  (let [config (into (into (vector number-input-neurons) vector-of-numbers-hidden-neurons) (vector number-output-neurons))
        layers-count (count config)
        tmp1 (take (dec (count config)) config)
        tmp2 (take-last (dec (count config)) config)
        layers (for [x (take (count (map vector tmp1 tmp2)) (map vector tmp1 tmp2))]
                 (conj (#(create-random-matrix-by-gaussian (inc (second x)) (inc (first x)) ))))
        layer-out (for [x tmp2]
                       (conj (#(create-null-matrix 1 (inc x)))))
        -   (doseq [x (range (dec layers-count))]
            (prepare-last-col (nth layers x)))]
      (->Neuronetwork layers
                      config
                      layer-out)
    )
  )

(defn create-temp-record
  "create temp record for calculations"
  [network input-mtx]
  (let [input-vec-dim (ncols input-mtx)
        net-input-dim (first (:config network))
        net-output-dim (last (:config network))
        tmp1 (take (dec (count (:config network))) (:config network))
        tmp2 (take-last (dec (count (:config network))) (:config network))
        layers-output (for [x tmp2]
                      (conj (#(create-null-matrix (inc x) input-vec-dim ))))
        layers-output-only (vec (map #(get-output-matrix %) layers-output))
        trans-weights (vec (map #(trans (get-weights-matrix %)) (:layers network)))

        temp-all-vector-o-gradients  (fge net-output-dim input-vec-dim (repeat net-output-dim 0))
        temp-all-vector-o-gradients2 (fge net-output-dim input-vec-dim (repeat net-output-dim 0))
        temp-all-vector-vector-h-gradients (vec (for [x tmp2] (fge x input-vec-dim (repeat x 0))))

        temp-vector-o-gradients  (fge net-output-dim 1 (repeat net-output-dim 0))
        temp-vector-o-gradients2 (fge net-output-dim 1 (repeat net-output-dim 0))
        temp-vector-vector-h-gradients (vec (for [x tmp2] (fge x 1 (repeat x 0))))
        temp-vector-matrix-delta (vec (concat (for [x (take (dec (count (map vector tmp1 tmp2))) (map vector tmp1 tmp2))]
                                                (conj (#(create-null-matrix (first x) (second x)))))
                                              (vector (create-null-matrix (first (last (map vector tmp1 tmp2)))
                                                                          (second (last (map vector tmp1 tmp2)))))))
        temp-vector-matrix-delta-biases (vec (for [x tmp2] (fge x 1 (repeat x 0))))
        temp-prev-delta-vector-matrix-delta (vec (concat (for [x (take (dec (count (map vector tmp1 tmp2))) (map vector tmp1 tmp2))]
                                                           (conj (#(create-null-matrix (first x) (second x)))))
                                                         (vector (create-null-matrix (first (last (map vector tmp1 tmp2)))
                                                                                     (second (last (map vector tmp1 tmp2)))))))
        temp-vector-matrix-delta-momentum (vec (concat (for [x (take (dec (count (map vector tmp1 tmp2))) (map vector tmp1 tmp2))]
                                                         (conj (#(create-null-matrix (first x) (second x)))))
                                                       (vector (create-null-matrix (first (last (map vector tmp1 tmp2)))
                                                                                   (second (last (map vector tmp1 tmp2)))))))
        ]
    (->Tempvariable layers-output
                    layers-output-only
                    trans-weights
                    temp-all-vector-o-gradients
                    temp-all-vector-o-gradients2
                    temp-all-vector-vector-h-gradients
                    temp-vector-o-gradients
                    temp-vector-o-gradients2
                    temp-vector-vector-h-gradients
                    temp-vector-matrix-delta
                    temp-vector-matrix-delta-biases
                    temp-prev-delta-vector-matrix-delta
                    temp-vector-matrix-delta-momentum
                    )
    )
  )

(defn feed-forward
  "feed forward propagation"
  [network input-mtx temp-variables]
  (let [input-vec-dim (mrows input-mtx)
        number-of-inputs (ncols input-mtx)
        net-input-dim (first (:config network))
        tmp2 (take-last (dec (count (:config network))) (:config network))
        number-of-layers (dec (count (:config network)))
        layers (:layers network)
        layers-output (:layers-output temp-variables)
        ]
    (if (= input-vec-dim (inc net-input-dim))
      (do
         (layer-output input-mtx (nth layers 0) (nth layers-output 0) tanh!)
         (doseq [y (range 0 (- number-of-layers 1))]
         (layer-output (nth layers-output y) (nth (:layers network) (inc y)) (nth layers-output (inc y)) tanh!) )
        )
      (throw (Exception. (str "Input dimmensions is not correct")))
      )
    )
  )

(defn xavier-initialization-update
  [network]
  (let [
        config (:config network)
        tmp1 (take (dec (count config)) config)
        tmp2 (take-last (dec (count config)) config)
        layer-neurons (map vector tmp1 tmp2)]
    (do
      ;; prepare weights for hidden layers
      (doseq [x (range (count layer-neurons))]
        (scal! (Math/sqrt (/ 2 (+ (first (nth layer-neurons x)) (second (nth layer-neurons x)))))
               (submatrix (nth (:layers network) x) (dec (mrows (nth (:layers network) x))) (ncols (nth (:layers network) x))) ))
      )
    )
  )

(defn copy-matrix-delta
  "save delta matrix for momentum"
  [temp-vars]
  (let [delta-matrix (:temp-vector-matrix-delta temp-vars)
        prev-delta-matrix (:temp-prev-delta-vector-matrix-delta temp-vars)
        layers-count (count delta-matrix)]
    (for [x (range layers-count)]
      (copy! (nth delta-matrix x) (nth prev-delta-matrix x)))
    )
  )

(defn backpropagation
  "learn network with one input vector"
  [network inputmtx no targetmtx temp-vars speed-learning alpha]
  (let [hidden-layers (take (dec (count (:layers network))) (:layers network))
        output-layer (last (:layers network))
        layers (:layers network)
        config (:config network)
        trans-weights (:trans-weights temp-vars)
        temp-matrix (:layers-output temp-vars)
        temp-matrix-only (:layers-output-only temp-vars)

        temp-all-vector-o-gradients (last (:temp-all-vector-vector-h-gradients temp-vars))
        temp-all-vector-o-gradients2 (:temp-all-vector-o-gradients2 temp-vars)
        temp-all-vector-vector-h-gradients (:temp-all-vector-vector-h-gradients temp-vars)

        temp-vector-o-gradients (last (:temp-vector-vector-h-gradients temp-vars))
        temp-vector-o-gradients2 (:temp-vector-o-gradients2 temp-vars)
        temp-vector-vector-h-gradients (:temp-vector-vector-h-gradients temp-vars)
        input (submatrix inputmtx 0 0 (inc (first (:config network))) 1)
        ;; target (submatrix targetmtx 0 no (last (:config network)) 1)
        temp-matrix-input (conj temp-matrix input)
        ]
    (do
      (entry! temp-vector-o-gradients 0)
      (entry! (:temp-all-vector-o-gradients temp-vars) 0)
      (entry! (:temp-all-vector-o-gradients2 temp-vars) 0)
      (feed-forward network inputmtx temp-vars)

      (if (not (= alpha 0))
        (copy-matrix-delta network))

      ;; calculate output gradients
      (axpy! -1 (last temp-matrix-only) temp-all-vector-o-gradients)

      (axpy! 1 targetmtx temp-all-vector-o-gradients)
      (dtanh! (last temp-matrix-only) temp-all-vector-o-gradients2)
      (mul! temp-all-vector-o-gradients2 temp-all-vector-o-gradients temp-all-vector-o-gradients)

      ;; calculate hidden gradients
      (doseq [x (range (- (count temp-matrix) 1) 0 -1)]
        (let [temp-h-gradients (nth (:temp-all-vector-vector-h-gradients temp-vars) x)
              temp-h-gradients-prev (nth (:temp-all-vector-vector-h-gradients temp-vars) (dec x))]
          (do
            ;; (mm! 1.0 (trans (get-weights-matrix (nth layers x))) temp-h-gradients
            (mm! 1.0 (nth trans-weights x) temp-h-gradients
                 0.0 temp-h-gradients-prev)
            (mul! (nth temp-matrix-only (dec x)) temp-h-gradients-prev))))

      ;; calculate average gradients
      (doseq [layer-grad (range (count (:temp-vector-vector-h-gradients temp-vars)))]
        (let [temp-unit-mtx (submatrix unit-matrix 0 0 (ncols inputmtx) 1)
              curr-all-gradients (nth (:temp-all-vector-vector-h-gradients temp-vars) layer-grad)
              curr-avg-gradients (nth (:temp-vector-vector-h-gradients temp-vars) layer-grad)]

          (mm! 1 curr-all-gradients temp-unit-mtx 0 curr-avg-gradients)
          (scal! (/ 1 (ncols curr-all-gradients)) curr-avg-gradients)

          )
        )

      ;; calculate average outputs
      (doseq [layer-grad (range (count (:layers-output-only temp-vars)))]
        (let [temp-unit-mtx (submatrix unit-matrix 0 0 (ncols (nth (:layers-output-only temp-vars) layer-grad)) 1)
              curr-outputs (nth (:layers-output-only temp-vars) layer-grad)
              curr-avg-outputs (submatrix (nth (:layers-output-only temp-vars) layer-grad) 0 0 (mrows curr-outputs) 1)]
          (mm! 1 curr-outputs temp-unit-mtx 0 curr-avg-outputs)
          (scal! (/ 1 (ncols curr-outputs)) curr-avg-outputs)
          )
        )

      ;; calculate delta for weights
      (doseq [row_o (range (- (count (:config network)) 2) -1 -1)]
        (let [layer-out-vector (col (get-output-matrix (nth temp-matrix-input row_o)) 0)
              cols-num (nth (take-last (dec (count config)) config) row_o)]
          (doseq [x (range cols-num)]
            (axpy! speed-learning layer-out-vector
                   (col (nth (:temp-vector-matrix-delta temp-vars) row_o) x))
            )))

      ;;      (doseq [row_o (range (- (count (:config network)) 2) -1 -1)]
      ;;      (axpy! (submatrix unit-matrix 0 0
      ;;                        (mrows (nth (:temp-vector-matrix-delta temp-vars) row_o))
      ;;                        (ncols (nth (:temp-vector-matrix-delta temp-vars) row_o))
      ;;      )
      ;;             (nth (:temp-vector-matrix-delta temp-vars) row_o)
      ;;      )
      ;;(scal! speed-learning (nth (:temp-vector-matrix-delta temp-vars) row_o))
      ;;      )

      (doseq [layer-grad (range (count (:temp-vector-vector-h-gradients temp-vars)))]
        (let []
          (doseq [x (range (nth (take-last (dec (count config)) config) layer-grad))]
            (scal! (entry (row (nth (:temp-vector-vector-h-gradients temp-vars) layer-grad) x) 0)
                   (col (nth (:temp-vector-matrix-delta temp-vars) layer-grad) x)
                   )))

        (axpy! (nth (:temp-vector-matrix-delta temp-vars) layer-grad)
               (nth trans-weights layer-grad)
               )

        ;; update biases
        (mul! (nth (:temp-vector-vector-h-gradients temp-vars) layer-grad)
              (get-bias-vector (nth layers layer-grad))
              (nth (:temp-vector-matrix-delta-biases temp-vars) layer-grad)
              )

        (scal! speed-learning (nth (:temp-vector-matrix-delta-biases temp-vars) layer-grad))
        (axpy! (nth (:temp-vector-matrix-delta-biases temp-vars) layer-grad)
               (get-bias-vector (nth layers layer-grad)))

        ;; momentum, if alpha <> 0
        (if (not (= alpha 0))
          (axpy! alpha (nth (:temp-prev-delta-vector-matrix-delta temp-vars) layer-grad)
                 (nth trans-weights layer-grad)
                 ))
        )

      ))
  )



(defn backpropagation-backup
  "learn network with one input vector"
  [network inputmtx no targetmtx temp-vars speed-learning alpha]
  (let [hidden-layers (take (dec (count (:layers network))) (:layers network))
        output-layer (last (:layers network))
        layers (:layers network)
        config (:config network)
        trans-weights (:trans-weights temp-vars)
        temp-matrix (:layers-output temp-vars)
        temp-matrix-only (:layers-output-only temp-vars)
        temp-vector-o-gradients (last (:temp-vector-vector-h-gradients temp-vars))
        temp-vector-o-gradients2 (:temp-vector-o-gradients2 temp-vars)
        temp-vector-vector-h-gradients (:temp-vector-vector-h-gradients temp-vars)
        input (submatrix inputmtx 0 no (inc (first (:config network))) 1)
        target (submatrix targetmtx 0 no (last (:config network)) 1)
        temp-matrix-input (conj temp-matrix input)]
    (do
      (entry! temp-vector-o-gradients 0)
      (entry! (:temp-vector-o-gradients2 temp-vars) 0)
      (feed-forward network input temp-vars)

      (if (not (= alpha 0))
        (copy-matrix-delta network)
        )

      ;; calculate output gradients
      (axpy! -1 (last temp-matrix-only) temp-vector-o-gradients)

      (axpy! 1 target temp-vector-o-gradients)
      (dtanh! (last temp-matrix-only) temp-vector-o-gradients2)
      (mul! temp-vector-o-gradients2 temp-vector-o-gradients temp-vector-o-gradients)

      ;; calculate hidden gradients
      (doseq [x (range (- (count temp-matrix) 1) 0 -1)]
        (let [temp-h-gradients (nth (:temp-vector-vector-h-gradients temp-vars) x)
              temp-h-gradients-prev (nth (:temp-vector-vector-h-gradients temp-vars) (dec x))]
          (do
            ;; (mm! 1.0 (trans (get-weights-matrix (nth layers x))) temp-h-gradients
            (mm! 1.0 (nth trans-weights x) temp-h-gradients
                 0.0 temp-h-gradients-prev)
            (mul! (nth temp-matrix-only (dec x)) temp-h-gradients-prev))))

      ;; calculate delta for weights
      ;;(doseq [row_o (range (- (count (conj temp-matrix input)) 2) -1 -1)]
      (doseq [row_o (range (- (count (:config network)) 2) -1 -1)]
        (let [layer-out-vector (col (get-output-matrix (nth temp-matrix-input row_o)) 0)
              ;;cols-num (ncols (nth (:temp-vector-matrix-delta temp-vars) row_o))]
              cols-num (nth (take-last (dec (count config)) config) row_o)]
          (doseq [x (range cols-num)]
            (axpy! speed-learning layer-out-vector
                   (col (nth (:temp-vector-matrix-delta temp-vars) row_o) x))
            )))

      (doseq [layer-grad (range (count (:temp-vector-vector-h-gradients temp-vars)))]

        (let []
          ;; (doseq [x (range (mrows (nth (:temp-vector-vector-h-gradients temp-vars) layer-grad)))]
          (doseq [x (range (nth (take-last (dec (count config)) config) layer-grad))]
            (scal! (entry (row (nth (:temp-vector-vector-h-gradients temp-vars) layer-grad) x) 0)
                   (col (nth (:temp-vector-matrix-delta temp-vars) layer-grad) x)
                   )))

        (axpy! (nth (:temp-vector-matrix-delta temp-vars) layer-grad)
               ;;  (trans (get-weights-matrix (nth layers layer-grad)))
               (nth trans-weights layer-grad)
               )

        ;; update biases
               (mul! (nth (:temp-vector-vector-h-gradients temp-vars) layer-grad)
                   (get-bias-vector (nth layers layer-grad))
                      (nth (:temp-vector-matrix-delta-biases temp-vars) layer-grad)
                )

                (scal! speed-learning (nth (:temp-vector-matrix-delta-biases temp-vars) layer-grad))
                (axpy! (nth (:temp-vector-matrix-delta-biases temp-vars) layer-grad)
                       (get-bias-vector (nth layers layer-grad)) )

        ;; momentum, if alpha <> 0
        (if (not (= alpha 0))
          (axpy! alpha (nth (:temp-prev-delta-vector-matrix-delta temp-vars) layer-grad)
                 ;; (trans (get-weights-matrix (nth layers layer-grad)))
                 (nth trans-weights layer-grad)
                 )
          )
        )

      )
    )
  )



(defn predict
  "feed forward propagation - prediction consumptions for input matrix"
  [network input-mtx temp-variables]
  (let [net-input-dim  (first (:config network))
        number-of-layers (dec (count (:config network)))
        input-vec-dim  (mrows input-mtx)
        number-of-inputs (ncols input-mtx)
        output-vec-rows (last (:config network))]

    (if (= input-vec-dim (inc net-input-dim))
      ;;(submatrix (feed-forward network input-mtx temp-variables) 0 0 number-of-inputs output-vec-rows)
      (do
        (feed-forward network input-mtx temp-variables)
        (submatrix (nth (:layers-output temp-variables) (dec number-of-layers)) 0 0 1 number-of-inputs)
        )

      (throw (Exception.
               (str "Error. Input dimensions is not correct. Expected dimension is: " net-input-dim)))
      ))
  )

(defn evaluate-original
  "evaluate restored values"
  [output-vec target-vec]
  (div (abs (axpy -1 target-vec output-vec)) target-vec)
  )

(defn evaluate-original-mape
  "MAPE calculations"
  [error-vec]
  (* (/ (sum error-vec) (dim error-vec)) 100)
  )

(defn learning-rate
  "learninig rate decay algorithm"
  [start-speed-learning decay-rate epoch-num]
  ;;(/ start-speed-learning (+ 1 (* decay-rate epoch-num)))
  (* start-speed-learning (Math/pow 0.95 epoch-num))
  )

(defn train-network
  "train network with input/target vectors"
  [network input-mtx target-mtx epoch-count mini-batch-size temp-vars3 speed-learning alpha]
  (let [line-count (ncols input-mtx)
        col-count (mrows input-mtx)
        tcol-count (mrows target-mtx)
        temp-vars2 (create-temp-record network (:normalized-matrix input-test-dataset))
        mini-batch-seg (conj (map #(vector %1 %2) (range 0 line-count mini-batch-size) (range mini-batch-size line-count mini-batch-size))
                      [(reduce max (range 0 line-count mini-batch-size)) line-count])
        ]
    (doseq [y (range epoch-count)]
       (doseq [x (shuffle (range (count mini-batch-seg)))]
        (let [prvi (first (nth mini-batch-seg x))
              drugi (second (nth mini-batch-seg x))
              input-segment (submatrix input-mtx 0 prvi col-count (- drugi prvi))
              target-segment (submatrix target-mtx 0 prvi tcol-count (- drugi prvi))
              temp-vars (create-temp-record network input-segment)]
          (backpropagation network input-segment 1 target-segment temp-vars speed-learning alpha)
          ))

      (let [os (mod y 1)]
        (if (= os 0)
          (let [mape-value
                (evaluate-original-mape
                  (evaluate-original
                    (restore-output-vector target-test-dataset (predict network (:normalized-matrix input-test-dataset) temp-vars2) 0)
                    (restore-output-vector target-test-dataset (:normalized-matrix target-test-dataset) 0)
                    ))]
            (do
              (println y ": " mape-value)
              (println "---------------------")
              (write-file "mini_batch_016.csv" (str y "," mape-value "\n"))

              ))))
      )))

(defn load-network-config
  "get network config from file file"
  [filename]
  (let [c-index (.indexOf (string/split (slurp (str "resources/" filename)) #"\n") "CONFIGURATION")
        l-index (.indexOf (string/split (slurp (str "resources/" filename)) #"\n") "LAYERS")]
    (map read-string (get (vec (map #(string/split % #",")
                                    (take 1 (nthnext
                                              (string/split
                                                (slurp (str "resources/" filename)) #"\n") (inc c-index)))))0))
    )
  )

(defn load-network-layers
  "get a output part of data from file"
  [filename x]
  (let [o-index (.indexOf (string/split (slurp (str "resources/" filename)) #"\n") (str "LAYER," (inc x)))
        e-index (.indexOf (string/split (slurp (str "resources/" filename)) #"\n") (str "LAYER," (+ x 2)))
        e-index2 (.indexOf (string/split (slurp (str "resources/" filename)) #"\n") "END")
        ]

    (if (= e-index -1)
      (map #(string/split % #",")
           (take (dec (- e-index2 o-index))
                 (nthnext
                   (string/split
                     (slurp (str "resources/" filename)) #"\n") (inc o-index))))
      (map #(string/split % #",")
           (take (dec (- e-index o-index))
                 (nthnext
                   (string/split
                     (slurp (str "resources/" filename)) #"\n") (inc o-index))))
      )
    ))


 (defn create-network-from-file
  "create new neural network and load state from file"
  [filename]
  (let [config (vec (load-network-config filename))
        tmp1 (take (dec (count config)) config)
        tmp2 (drop 1 config)
        layers (let [x (take (count (map vector tmp1 tmp2)) (map vector tmp1 tmp2))]
                        (for [y (range (count (map vector tmp1 tmp2)))]
                          (conj
                            (trans
                              (fge (inc (first (nth x y))) (inc (second (nth x y))) (reduce into [] (map #(map parse-float %)
                                                                                                         (load-network-layers filename y))))
                              )
                            ))
                        )
        layer-out (for [x tmp2]
                    (conj (#(create-null-matrix 1 (inc x)))))
    ]

    (->Neuronetwork layers
                    config
                    layer-out)
    )
  )