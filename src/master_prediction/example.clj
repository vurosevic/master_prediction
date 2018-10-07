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


(def mreza-nn (atom (create-network-gaussian 64 [80 80 80] 1)))
(def temp-variables2 (atom (create-temp-record @mreza-nn input-730-b)))
(xavier-initialization-update @mreza-nn)

(feed-forward @mreza-nn input-730-b temp-variables)

(mrows input-730-b)
(first (:config @mreza-nn))

(layer-output (:normalized-matrix norm-input-730) (trans (nth (:layers @mreza-nn) 0))
              (nth (:layers-output @temp-variables2) 0) tanh!)

(:normalized-matrix norm-input-730)
(:layers-output @temp-variables2)

(nth (:layers @mreza-nn) 0)
(trans (nth (:layers @mreza-nn) 0))
(trans (:normalized-matrix norm-input-730))

(nth (:layers-output @temp-variables2) 0)

(:layers @mreza-nn)
(:layers-output @temp-variables2)

(ncols input-730-b)

(nth (:layers @mreza-nn) 0)
