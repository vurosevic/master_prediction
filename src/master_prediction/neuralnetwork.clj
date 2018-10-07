(ns ^{:author "Vladimir Urosevic"}
master-prediction.neuralnetwork
  (:require [master-prediction.data :refer :all]
            [uncomplicate.neanderthal.core :refer :all]
            [uncomplicate.neanderthal.vect-math :refer :all]
            [uncomplicate.neanderthal.native :refer :all]
            [clojure.string :as string]
            [clojure.core :as core]))

(defrecord Tempvariable [
                         layers-output              ;; output for layers

                         ;;                         temp-vector-o-gradients             ;; matrix, row=1, output gradient, dim number of output neurons
                         ;;                         temp-vector-o-gradients2            ;; matrix, row=1, output gradient, dim number of output neurons

                         ;;                         temp-vector-vector-h-gradients      ;; output gradient, dim number of output neurons

                         ;;                         temp-matrix-gradients               ;; gradients for hidden layers, vectors by layer
                         ;;                         temp-vector-matrix-delta            ;; delta weights for layers
                         ;;                         temp-vector-matrix-delta-biases     ;; delta biases for layers
                         ;;                         temp-prev-delta-vector-matrix-delta ;; previous delta vector matrix layers
                         ;;                         temp-vector-matrix-delta-momentum   ;; delta weights for layers - momentum
                         ])

(defrecord Neuronetwork [
                         layers                              ;; vector of layers (hiddens and output weights and biases)
                         config                              ;; vector, numbers of neurons by layer
                         layers-output                       ;; vector of matrix, outputs from layers
                         ])

(def max-dim 4096)

(def unit-vector (dv (replicate max-dim 1)))
(def unit-matrix (dge max-dim max-dim (repeat 1)))

(defn prepare-unit-vector
  "preparing unit vector for other calculations"
  [n]
  (if (<= n max-dim)
    (subvector unit-vector 0 n)
    (dv [0])))

(import '(java.util Random))
(def normals
  (let [r (Random.)]
    (map #(/ % 4.8) (take 500000 (repeatedly #(-> r .nextGaussian (* 0.9) (+ 1.0)))))
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
    (dge dim-y dim-x (nthrest normals @temp-current-val))
    ))

(defn create-null-matrix
  "Initialize a layer"
  [dim-y dim-x]
  (do
    (if (> dim-y max-dim)
      (throw (Exception. (str "Error. Max number of neurons is " max-dim))))
    (if (> dim-x max-dim)
      (throw (Exception. (str "Error. Max number of neurons is " max-dim))))
    (dge dim-y dim-x))
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

(defn prepare-last-col2
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

(defn prepare-last-col
  "Sets last column to [0 0 0 ... 1]"
  [input-matrix]
  (let [last-col (ncols input-matrix)
        col (col input-matrix (dec last-col))]
    (do
      (scal! 0 col)
      (entry! col (dec (dim col)) 1)
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
                 (conj (#(create-random-matrix-by-gaussian (inc (first x)) (inc (second x))))))
        layer-output (for [x tmp2]
                       (conj (#(create-null-matrix (inc x) 1))))
        -   (doseq [x (range (dec layers-count))]
            (prepare-last-col (nth layers x)))]
      (->Neuronetwork layers
                      config
                      layer-output)
    )
  )

(defn create-temp-record
  "create temp record for calculations"
  [network input-mtx]
  (let [input-vec-dim (mrows input-mtx)
        net-input-dim (first (:config network))
        tmp2 (take-last (dec (count (:config network))) (:config network))
        layers-output (for [x tmp2]
                      (conj (#(create-null-matrix (inc x) (ncols input-mtx)))))]

    (->Tempvariable layers-output)
    )
  )

(defn feed-forward
  "feed forward propagation"
  [network input-mtx temp-variables]
  (let [input-vec-dim (mrows input-mtx)
        net-input-dim (first (:config network))
        tmp2 (take-last (dec (count (:config network))) (:config network))
        number-of-layers (dec (count (:config network)))
        layers (:layers network)
        layers-output (:layers-output temp-variables)
        ]
    (if (= input-vec-dim (inc net-input-dim))
      (do

        (layer-output input-mtx (nth layers 0) (nth layers-output 0) tanh!)
        ;; (doseq [y (range 0 (- number-of-layers 1))]
        ;;  (layer-output (nth layers-output y) (trans (nth (:layers network) (inc y))) (nth layers-output (inc y)) tanh!) )

        ;;(layer-output (nth temp-matrix (- number-of-layers 2)) (trans (:output-layer network)) (nth temp-matrix (- number-of-layers 1)) tanh!)
        ;; (nth layers-output (dec number-of-layers))
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
               (submatrix (nth (:layers network) x) (dec (mrows (nth (:layers network) x))) (ncols (nth (:layers network) x)) ) ))
      )
    )

  )