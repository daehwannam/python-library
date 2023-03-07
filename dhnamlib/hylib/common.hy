
(import [hy [HySymbol HyExpression HyList HyDict HySet]]
        [hy.models [HyObject HySequence HyString HyInteger HyFloat]])
(import hy.models)

(import [hy.contrib.hy-repr [hy-repr]])
;; (import [.pyeval [pyeval]])


;; Hy structures
(defn hysymb [obj]
  ;; 'name can be used to get string of a keyword
  ;; (cut (str kw) 1) == (name kw)
  (cond [(keyword? obj) (HySymbol (name obj))]
        [True (HySymbol obj)]))


(defn hyexpr [&rest args]
  (HyExpression args))


(defn hylist [&rest args]
  (HyList args))


(defn hyset [&rest args]
  (HySet args))


(defn hyexpr? [arg]
  (instance? HyExpression arg))


(defn hylist? [arg]
  (instance? HyList arg))


(defn hystring? [arg]
  (instance? HyString arg))


;; (defn hyint? [arg]
;;   (instance? Hyinteger arg))


;; (defn hyfloat? [arg]
;;   (instance? HyFloat arg))


(defn hyobject? [arg]
  (instance? HyObject arg))


(defn hysequence? [arg]
  (instance? HySequence arg))


(defn annotation? [obj]
  (and (instance? HyExpression obj) (= (len obj) 2) (= (get obj 0) 'annotate*)))


(defn tuple-repr? [arg]
  (and (hyexpr? arg) (not (empty? arg)) (= (first arg) ',)))

;; printing
(defn hy-print [&rest args &kwargs kargs]
  (print #* (map (fn [x] (cond [(instance? str x) x]
                               [True (hy-repr x)]))
                 args) #** kargs))


(defn get-expr-str [expr]
  (assert (instance? HyObject expr))
  (cut (hy-repr expr) 1))

;; sequence
;; (defn filter-recursively [func coll]
;;   ((type coll) (remove none?
;;                        (map (fn [x] (cond [(iterable? x) (filter-recursively func x)]
;;                                           [(func x) x]
;;                                           [True None]))
;;                             coll)

(defn filter-recursively [func coll]
  (defn get-gen []
    (for [x coll]
      (cond
        [(iterable? x) (yield ((type x) (filter-recursively func x)))]
        [(func x) (yield x)])))
  ((type coll) (get-gen)))


(defn filter-hyexpr-recursively [func coll]
  (HyExpression (remove none?
                       (map (fn [x] (cond [(hyexpr? x) (filter-hyexpr-recursively func x)]
                                          [(func x) x]
                                          [True None]))
                            coll))))

(defn replace-symbol-recursively [coll old-symbol new-symbol]
  (defn get-gen []
    (for [x coll]
      (cond
        [(symbol? x)
         (if (= x old-symbol)
             (yield new-symbol)
             (yield x))]
        [(iterable? x) (yield ((type x) (replace-symbol-recursively
                                          x old-symbol new-symbol)))]
        [(func x) (yield x)])))
  ((type coll) (get-gen)))

  ;; (cond
  ;;   [(hyexpr? coll)
  ;;     (HyExpression (map (fn [x] (filter-hyexpr-recursively func x)) coll))]
  ;;   [(func coll)
  ;;    coll]
  ;;   [True 100]))


;; (defn parse-pairs [pair-expr &kwonly [leftfn identity] [rightfn identity]]
;;   (assert (even? (len pair-expr)))
;;   (HyExpression (gfor [k v] (partition pair-expr :fillvalue None) (hylist (leftfn k) (rightfn v)))))


(do
  (setv hy-py-seq-cls-dict
        (dict (partition
                [hy.models.HyList list
                 hy.models.HyDict dict
                 hy.models.HySet set])))

  (setv hy-py-nonseq-cls-dict
        (dict (partition
                [hy.models.HyString str
                 hy.models.HyInteger int
                 hy.models.HyFloat float])))

  (defn expr2data [expr]
    (cond
      [(tuple-repr? expr)
       (tuple (map expr2data (rest expr)))]
      [(hysequence? expr)
       (setv py-seq-cls
             (or (.get hy-py-seq-cls-dict (type expr))
                 (cond [(instance? HyExpression expr) HyExpression]
                       [True (raise (Exception "Unknown sequence type"))])))
       (py-seq-cls (map expr2data expr))]
      [True
       (setv py-nonseq-cls (.get hy-py-nonseq-cls-dict (type expr)))
       (if (none? py-nonseq-cls)
           (cond
             [(instance? HySymbol expr)
              (cond
                [(= expr 'None) None]
                [(= expr 'True) True]
                [(= expr 'False) False]
                [True expr])]
             [True
              expr  ; it would be a data in itself
              #_(raise (Exception "Unknown non-sequence type"))])
           (py-nonseq-cls expr))]))
  )

(defn weak-mangle [text]
  ; instead of 'mangle which collapse dot(.) sign
  (.replace text "-" "_"))
