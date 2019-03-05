(ns ^{:author "Vladimir Urosevic"}
master-prediction.neuralnetwork
  (:require [master-prediction.data :refer :all]
            [uncomplicate.neanderthal.core :refer :all]
            [uncomplicate.neanderthal.vect-math :refer :all]
            [uncomplicate.neanderthal.native :refer :all]
            [clojure.string :as string]
            [clojure.core :as core]))

(defrecord Tempvariable [
                         layers-output                      ;; output for layers + 1 for biases for next leyer
                         layers-output-only                 ;; output for layers

                         trans-weights                      ;; trans weights matrixes

                         temp-all-vector-o-signals          ;; matrix, row=1, output gradient, dim number of output neurons
                         temp-all-vector-o-signals2         ;; matrix, row=1, output gradient, dim number of output neurons
                         temp-all-vector-vector-h-signals   ;; output gradient, dim number of output neurons

                         temp-vector-o-gradients            ;; matrix, row=1, output gradient, dim number of output neurons
                         temp-vector-o-gradients2           ;; matrix, row=1, output gradient, dim number of output neurons
                         temp-vector-vector-h-gradients     ;; output gradient, dim number of output neurons

                         temp-vector-matrix-delta           ;; delta weights for layers
                         temp-vector-matrix-gradient        ;; temp for calc weights for layers
                         temp-vector-matrix-delta-biases    ;; delta biases for layers
                         temp-vector-matrix-biases          ;; temp for calc biases for layers
                         temp-prev-delta-vector-matrix-delta ;; previous delta vector matrix weights
                         temp-prev-vector-matrix-delta-biases ;; previous delta vector matrix biases
                         temp-vector-matrix-delta-momentum  ;; delta weights for layers - momentum
                         temp-vector-vector-h-signals-var1  ;; temp variables for signals calculating
                         temp-vector-vector-h-signals-var2  ;; temp variables for signals calculating
                         ])

(defrecord Neuronetwork [
                         layers                             ;; vector of layers (hiddens and output weights and biases)
                         config                             ;; vector, numbers of neurons by layer
                         ])

(def max-layers 20)
(def max-dim 4096)
(def unit-vector (dv (replicate max-dim 1)))
(def unit-matrix (fge max-dim max-dim (repeat 1)))
(def zero-matrix (fge max-dim max-dim (repeat 0)))

(def identity-mtx (fge max-dim max-dim (repeat 0)))
(entry! (dia identity-mtx) 1)

(defn prepare-zero-matrix
  "preparing identity matrix for other calculations NxN"
  [n]
  (if (<= n max-dim)
    (submatrix zero-matrix 0 0 n n)
    (fge 0 0))
  )

(defn prepare-identity-matrix
  "preparing identity matrix for other calculations NxN"
  [n]
  (if (<= n max-dim)
    (submatrix identity-mtx 0 0 n n)
    (fge 0 0))
  )

(defn prepare-unit-vector
  "preparing unit vector for other calculations"
  [n]
  (if (<= n max-dim)
    (subvector unit-vector 0 n)
    (dv [0])))

(import '(java.util Random))
(def normals
  (let [r (Random.)]                                        ;; 4.7
    (map #(/ % 4.7) (take (* max-layers (* max-dim max-dim)) (repeatedly #(-> r .nextGaussian (* 0.9) (+ 1.0)))))
    ))

(def temp-current-val (atom 0))
(defn create-random-matrix
  "Initialize a layer by gaussian"
  [dim-y dim-x]
  (do
    (if (> dim-y max-dim)
      (throw (Exception. (str "Error. Max number of neurons is " max-dim))))
    (if (> dim-x max-dim)
      (throw (Exception. (str "Error. Max number of neurons is " max-dim))))
    (let [matrix (fge dim-y dim-x (nthrest normals @temp-current-val))
          - (reset! temp-current-val (+ @temp-current-val (* dim-x dim-y)))]
      matrix)
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
  (o-func (submatrix result 0 0 (dec (mrows result)) (ncols result))))

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
      )))

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
    ))

(defn get-bias-vector
  "return biases from layer"
  [layer]
  (let [num-rows (dec (mrows layer))
        num-cols (dec (ncols layer))]
    (submatrix layer 0 num-cols num-rows 1)))

(defn set-biases-value2
  "initialize biases for neural network to value"
  [network value]
  (let [layers (:layers network)
        num-layers (count layers)
        biases (vec (map #(col (get-bias-vector %) 0) layers))]
    (doseq [x (range (count biases))]
      (entry! (nth biases x) value)
      )))

(defn set-biases-value
  "initialize biases for neural network to value"
  [layers value]
  (let [num-layers (count layers)
        biases (vec (map #(col (get-bias-vector %) 0) layers))]
    (doseq [x (range (count biases))]
      (entry! (nth biases x) value)
      )))

(defn xavier-initialization-update
  [layers config]
  (let [
        tmp1 (take (dec (count config)) config)
        tmp2 (take-last (dec (count config)) config)
        layer-neurons (map vector tmp1 tmp2)]
    (do
      ;; prepare weights for hidden layers
      (doseq [x (range (count layer-neurons))]
        (scal! (Math/sqrt (/ 2 (+ (first (nth layer-neurons x)) (second (nth layer-neurons x)))))
               (submatrix (nth layers x) (dec (mrows (nth layers x))) (ncols (nth layers x))))
        ))))

(defn create-network
  "create new neural network"
  [number-input-neurons vector-of-numbers-hidden-neurons number-output-neurons]
  (let [config (into (into (vector number-input-neurons) vector-of-numbers-hidden-neurons) (vector number-output-neurons))
        layers-count (count config)
        tmp1 (take (dec (count config)) config)
        tmp2 (take-last (dec (count config)) config)
        layers (for [x (take (count (map vector tmp1 tmp2)) (map vector tmp1 tmp2))]
               (conj (#(create-random-matrix (inc (second x)) (inc (first x))))))
        - (doseq [x (range (dec layers-count))]
            (prepare-last-col (nth layers x)))]
    - (xavier-initialization-update layers config)
    - (set-biases-value layers 0)
    (->Neuronetwork layers
                    config
                    )))

(defn create-temp-record
  "create temp record for calculations"
  [network input-mtx]
  (let [input-vec-dim (ncols input-mtx)
        net-input-dim (first (:config network))
        net-output-dim (last (:config network))
        tmp1 (take (dec (count (:config network))) (:config network))
        tmp2 (take-last (dec (count (:config network))) (:config network))
        layers-output (for [x tmp2]
                        (conj (#(create-null-matrix (inc x) input-vec-dim))))
        layers-output-only (vec (map #(get-output-matrix %) layers-output))
        trans-weights (vec (map #(trans (get-weights-matrix %)) (:layers network)))

        temp-all-vector-o-signals (fge net-output-dim input-vec-dim (repeat net-output-dim 0))
        temp-all-vector-o-signals2 (fge net-output-dim input-vec-dim (repeat net-output-dim 0))
        temp-all-vector-vector-h-signals (vec (for [x tmp2] (fge x input-vec-dim (repeat x 0))))

        temp-vector-o-gradients (fge net-output-dim 1 (repeat net-output-dim 0))
        temp-vector-o-gradients2 (fge net-output-dim 1 (repeat net-output-dim 0))
        temp-vector-vector-h-gradients (vec (for [x tmp2] (fge x 1 (repeat x 0))))
        temp-vector-matrix-delta (vec (concat (for [x (take (dec (count (map vector tmp1 tmp2))) (map vector tmp1 tmp2))]
                                                (conj (#(create-null-matrix (first x) (second x)))))
                                              (vector (create-null-matrix (first (last (map vector tmp1 tmp2)))
                                                                          (second (last (map vector tmp1 tmp2)))))))
        temp-vector-matrix-gradient (vec (concat (for [x (take (dec (count (map vector tmp1 tmp2))) (map vector tmp1 tmp2))]
                                                   (conj (#(create-null-matrix (first x) (second x)))))
                                                 (vector (create-null-matrix (first (last (map vector tmp1 tmp2)))
                                                                             (second (last (map vector tmp1 tmp2)))))))
        temp-vector-matrix-delta-biases (vec (for [x tmp2] (fge x 1 (repeat x 0))))
        temp-vector-matrix-biases (vec (for [x tmp2] (fge x 1 (repeat x 0))))
        temp-prev-delta-vector-matrix-delta (vec (concat (for [x (take (dec (count (map vector tmp1 tmp2))) (map vector tmp1 tmp2))]
                                                           (conj (#(create-null-matrix (first x) (second x)))))
                                                         (vector (create-null-matrix (first (last (map vector tmp1 tmp2)))
                                                                                     (second (last (map vector tmp1 tmp2)))))))
        temp-prev-vector-matrix-delta-biases (vec (for [x tmp2] (fge x 1 (repeat x 0))))
        temp-vector-matrix-delta-momentum (vec (concat (for [x (take (dec (count (map vector tmp1 tmp2))) (map vector tmp1 tmp2))]
                                                         (conj (#(create-null-matrix (first x) (second x)))))
                                                       (vector (create-null-matrix (first (last (map vector tmp1 tmp2)))
                                                                                   (second (last (map vector tmp1 tmp2)))))))

        temp-vector-vector-h-signals-var1 (for [x (take (count (map vector tmp1 tmp2)) (map vector tmp1 tmp2))]
                                            (conj (#(create-null-matrix (second x) (first x)))))
        temp-vector-vector-h-signals-var2 (vec (for [x tmp2] (fge 1 x (repeat x 0))))

        ]
    (->Tempvariable layers-output
                    layers-output-only
                    trans-weights
                    temp-all-vector-o-signals
                    temp-all-vector-o-signals2
                    temp-all-vector-vector-h-signals
                    temp-vector-o-gradients
                    temp-vector-o-gradients2
                    temp-vector-vector-h-gradients
                    temp-vector-matrix-delta
                    temp-vector-matrix-gradient
                    temp-vector-matrix-delta-biases
                    temp-vector-matrix-biases
                    temp-prev-delta-vector-matrix-delta
                    temp-prev-vector-matrix-delta-biases
                    temp-vector-matrix-delta-momentum
                    temp-vector-vector-h-signals-var1
                    temp-vector-vector-h-signals-var2
                    )))

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
          (layer-output (nth layers-output y) (nth (:layers network) (inc y)) (nth layers-output (inc y)) tanh!))
        )
      (throw (Exception. (str "Input dimmensions is not correct"))))))

(defn copy-matrix-delta
  "save delta matrix for momentum"
  [temp-vars]
  (let [delta-matrix (:temp-vector-matrix-delta temp-vars)
        delta-biases (:temp-vector-matrix-delta-biases temp-vars)
        prev-delta-matrix (:temp-prev-delta-vector-matrix-delta temp-vars)
        prev-delta-biases (:temp-prev-vector-matrix-delta-biases temp-vars)
        layers-count (count delta-matrix)]
    (for [x (range layers-count)]
      (do
        (copy! (nth delta-matrix x) (nth prev-delta-matrix x))
        (copy! (nth delta-biases x) (nth prev-delta-biases x))
        )
      )
    )
  )

(defn learning-once
  "learn network with one input vector"
  [network inputmtx targetmtx temp-vars speed-learning alpha]
  (let [layers (:layers network)
        trans-weights (:trans-weights temp-vars)
        temp-matrix-only (:layers-output-only temp-vars)

        temp-all-vector-o-signals (last (:temp-all-vector-vector-h-signals temp-vars))
        temp-all-vector-o-signals2 (:temp-all-vector-o-signals2 temp-vars)
        temp-all-vector-vector-h-signals (:temp-all-vector-vector-h-signals temp-vars)
        temp-vector-matrix-gradient (:temp-vector-matrix-gradient temp-vars)

        inputm (get-output-matrix inputmtx)]
    (do

      (entry! (:temp-all-vector-o-signals temp-vars) 0)
      (entry! (:temp-all-vector-o-signals2 temp-vars) 0)

      (doseq [dd (range (count (:temp-prev-delta-vector-matrix-delta temp-vars)))]
        (entry! (nth (:temp-vector-matrix-delta temp-vars) dd) 0)
        (entry! (nth (:temp-prev-delta-vector-matrix-delta temp-vars) dd) 0)
        (entry! (nth (:temp-vector-matrix-delta-biases temp-vars) dd) 0))

      (feed-forward network inputmtx temp-vars)

      (if (not (= alpha 0))
        (copy-matrix-delta network))

      ;; compute output node signals
      (axpy! targetmtx temp-all-vector-o-signals2)
      (axpy! -1 (last temp-matrix-only) temp-all-vector-o-signals2)
      (dtanh! (last temp-matrix-only) temp-all-vector-o-signals)
      (mul! temp-all-vector-o-signals2 temp-all-vector-o-signals temp-all-vector-o-signals)


      ;; compute other node signals
      (doseq [x (range (- (count layers) 1) 0 -1)]
        (let [temp-h-signals (nth (:temp-all-vector-vector-h-signals temp-vars) x)
              temp-h-signals-prev (nth (:temp-all-vector-vector-h-signals temp-vars) (dec x))
              wght (get-weights-matrix (nth layers x))
              temp-signal-var1 (nth (:temp-vector-vector-h-signals-var1 temp-vars) x)
              temp-signal-var2 (nth (:temp-vector-vector-h-signals-var2 temp-vars) (dec x))]
          (do
            (dtanh! (nth temp-matrix-only (dec x)) temp-h-signals-prev)
            (doseq [sx (range (ncols temp-h-signals))]
              (let [
                    scol (col temp-h-signals sx)
                    ident-mtx (prepare-zero-matrix (dim scol))
                    unit-mtx (submatrix unit-matrix 0 0 1 (dim scol))
                    prev-scol (col temp-h-signals-prev sx)]

                (entry! (dia ident-mtx) 0)
                (axpy! scol (dia ident-mtx))

                (mm! 1 ident-mtx wght 0 temp-signal-var1)
                (mm! 1 unit-mtx temp-signal-var1 0 temp-signal-var2)
                (mul! (row temp-signal-var2 0) prev-scol prev-scol)
                )))))

      ;;compute and accumulate hidden weight gradients using output signals
      (doseq [x (range (- (count layers) 1) 0 -1)]
        (let [signals (nth temp-all-vector-vector-h-signals x)
              col-sig (ncols signals)
              rows-sig (mrows signals)
              out-mtx (nth temp-matrix-only (dec x))
              rows-out (mrows out-mtx)
              temp-mtx-delta (nth (:temp-vector-matrix-delta temp-vars) x)]
          (doseq [x-inter (range col-sig)]
            (mm! 1 (submatrix out-mtx 0 x-inter rows-out 1) (trans (submatrix signals 0 x-inter rows-sig 1)) 1 temp-mtx-delta)
            )))

      ;; compute and accumulate input-hidden weight gradients
      (let [signals (nth temp-all-vector-vector-h-signals 0)
            col-sig (ncols signals)
            rows-sig (mrows signals)
            rows-out (mrows inputm)
            temp-mtx-delta (nth (:temp-vector-matrix-delta temp-vars) 0)]

        (doseq [x-inter (range col-sig)]
          (mm! 1 (submatrix inputm 0 x-inter rows-out 1) (trans (submatrix signals 0 x-inter rows-sig 1)) 1 temp-mtx-delta)
          ))

      ;;compute and accumulate bias gradients
      (doseq [x (range (count layers))]
        (let [signals (nth temp-all-vector-vector-h-signals x)
              col-sig (ncols signals)
              sig-vec (submatrix unit-matrix 0 0 col-sig 1)]
          (do
            (mm! 1 signals sig-vec 1 (nth (:temp-vector-matrix-delta-biases temp-vars) x))
            )))

      ;; update weights and biases
      (doseq [x (range (count layers))]
        ;; update weights
        (axpy! speed-learning (nth (:temp-vector-matrix-delta temp-vars) x)
               (nth trans-weights x))

        ;; update biases
        (axpy! speed-learning (nth (:temp-vector-matrix-delta-biases temp-vars) x)
               (get-bias-vector (nth layers x)))

        ;; momentum, if alpha <> 0
        (if (not (= alpha 0))
          (do
            (axpy! alpha (nth (:temp-prev-delta-vector-matrix-delta temp-vars) x)
                   (nth trans-weights x))

            (axpy! alpha (nth (:temp-prev-vector-matrix-delta-biases temp-vars) x)
                   (get-bias-vector (nth layers x)))
            ))))))

(defn predict
  "feed forward propagation - prediction consumptions for input matrix"
  [network input-mtx temp-variables]
  (let [net-input-dim (first (:config network))
        number-of-layers (dec (count (:config network)))
        input-vec-dim (mrows input-mtx)
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
      )))

(defn evaluate
  "absolute percentage error for output values"
  [output-vec target-vec]
  (div (abs (axpy -1 target-vec output-vec)) target-vec))

(defn evaluate-mape
  "MAPE calculations"
  [error-vec]
  (* (/ (sum error-vec) (dim error-vec)) 100))

(defn learning-rate
  "learninig rate decay algorithm"
  [start-speed-learning decay-rate epoch-num]
  (/ start-speed-learning (+ 1 (* decay-rate epoch-num)))

  ;; second way
  ;;(* start-speed-learning (Math/pow 0.95 epoch-num))
  )

(def early-stopping-value (atom 2))
(defn train-network
  "train network with input/target vectors"
  [network input-mtx target-mtx epoch-count mini-batch-size speed-learning alpha]
  (let [line-count (ncols input-mtx)
        col-count (mrows input-mtx)
        tcol-count (mrows target-mtx)
        temp-vars2 (create-temp-record network (:normalized-matrix input-test-dataset))
        mini-batch-seg (conj (map #(vector %1 %2) (range 0 line-count mini-batch-size) (range mini-batch-size line-count mini-batch-size))
                             [(reduce max (range 0 line-count mini-batch-size)) line-count])

        ffirst-seg (first (nth mini-batch-seg 0))
        fsecond-seg (second (nth mini-batch-seg 0))
        finput-segment (submatrix input-mtx 0 ffirst-seg col-count (- fsecond-seg ffirst-seg))
        ftemp-vars (create-temp-record network finput-segment)

        lfirst-seg (first (last mini-batch-seg))
        lsecond-seg (second (last mini-batch-seg))
        linput-segment (submatrix input-mtx 0 lfirst-seg col-count (- lsecond-seg lfirst-seg))
        ltemp-vars (create-temp-record network linput-segment)]
    (doseq [y (range epoch-count)]
      (doseq [x (shuffle (range (count mini-batch-seg)))]
        (let [first-seg (first (nth mini-batch-seg x))
              second-seg (second (nth mini-batch-seg x))
              input-segment (submatrix input-mtx 0 first-seg col-count (- second-seg first-seg))
              target-segment (submatrix target-mtx 0 first-seg tcol-count (- second-seg first-seg))
              ;;temp-vars (create-temp-record network input-segment)
              temp-vars (if (= x 0)
                          ftemp-vars
                          ltemp-vars)]
          (learning-once network input-segment target-segment temp-vars speed-learning alpha)
          ;;(learning-once network input-segment target-segment temp-vars (learning-rate speed-learning 0.85 y) alpha)
          ))

      (let [os (mod y 1)]
        (if (= os 0)
          (let [mape-value
                (evaluate-mape
                  (evaluate
                    (restore-output-vector target-test-dataset (predict network (:normalized-matrix input-test-dataset) temp-vars2))
                    (restore-output-vector target-test-dataset (:normalized-matrix target-test-dataset))
                    ))]
            (do
              (println y ": " mape-value)
              (if (< mape-value @early-stopping-value)
                (do
                  (save-network-to-file network "early-stopping-net-test.csv")
                  (reset! early-stopping-value mape-value)))
              (write-file "konvergencija_minibatch_test.csv" (str y "," mape-value "\n"))
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
                                                (slurp (str "resources/" filename)) #"\n") (inc c-index))))) 0))
    ))

(defn load-network-layers
  "get a output part of data from file"
  [filename x]
  (let [o-index (.indexOf (string/split (slurp (str "resources/" filename)) #"\n") (str "LAYER," (inc x)))
        e-index (.indexOf (string/split (slurp (str "resources/" filename)) #"\n") (str "LAYER," (+ x 2)))
        e-index2 (.indexOf (string/split (slurp (str "resources/" filename)) #"\n") "END")]
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
      )))


(defn create-network-from-file
  "create new neural network and load state from file"
  [filename]
  (let [config (vec (load-network-config filename))
        tmp1 (take (dec (count config)) config)
        tmp2 (drop 1 config)
        layers (let [x (take (count (map vector tmp1 tmp2)) (map vector tmp1 tmp2))]
                 (for [y (range (count (map vector tmp1 tmp2)))]
                   (conj
                     (fge (inc (second (nth x y))) (inc (first (nth x y)))
                          (reduce into [] (map #(map parse-float %)
                                               (load-network-layers filename y)))
                          ))))]
    (->Neuronetwork layers
                    config)))

(defn create-predict-file1
  [net input target filename]
  (let [temp-variables (create-temp-record net input)
        pred-values (restore-output-vector target (predict net input temp-variables))
        values (restore-output-vector target (:normalized-matrix target))
        count-values (dim pred-values)]
    (doseq [x (range count-values)]
      (write-file filename (str x "," (entry values x) "," (entry pred-values x) "\n"))
      )
    )
  )