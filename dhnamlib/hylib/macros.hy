
(defmacro gattr [obj attr]
  "example: (gattr [1 2 3] append)"
  ;; `(getattr ~obj (str '~attr)))
  (setv attr_str (mangle attr))
  `(getattr ~obj ~attr_str))


(defmacro sattr [obj attr val]
  ;; `(setattr ~obj (str '~attr) ~val))
  (setv attr_str (mangle attr))
  `(setattr ~obj ~attr_str ~val))


(deftag renew [expr]
  #[document[example: (do (setv x "a-b-c") #renew(.split x "-") x)]document]
  (setv [func var #* args] expr)
  `(setv ~var (~func ~var ~@args)))

;; (defmacro renew [func var &rest args]  ; defmacro version
;;   #[document[example: (do (setv x "a-b-c") (renew .split x "-"))]document]
;;   `(setv ~var (~func ~var ~@args)))


(defmacro assert* [expr &optional msg]
  (if msg
      `(do
         (import [hy.contrib.hy-repr [hy-repr]])
         (assert ~expr ~msg))
      `(do
         (import [hy.contrib.hy-repr [hy-repr]])
         (assert ~expr (hy-repr '~expr)))))


(defmacro! kwargs2variables [o!kwargs variables]
  `(do ~@(HyExpression (gfor variable variables
                             `(setv ~variable (get ~g!kwargs (mangle (str '~variable))))))))

(defmacro update-value [dic &rest args]
  (setv kvpairs (partition args :fillvalue None))
  (setv body (HyExpression (chain #* (*map (fn [x y] [(+ `(get ~dic) `(~x)) `~y])
                                           kvpairs))))
  (+ '(setv) body))


;; (deftag do-block [expr]
;;   "Example:
;;      #do-block
;;       (test-add (+ 1 2 None))
;;   "
;;   (setv name-symbol (first expr))
;;   (setv body (rest expr))

;;   `(try
;;      (do ~@body)
;;      (except [e Exception]
;;        (setv current-file-path (if (in "__file__" (globals))
;;                                    (do
;;                                      (import os)
;;                                      (os.path.abspath __file__))
;;                                    "<non-file>"))
;;        (print (.format "Exception occured in do-block '{}' of {}" (str '~name-symbol) current-file-path))
;;        (raise e))))


(defmacro do-block [name-symbol &rest body]
  "Example:
     (do-block
       test-add
       (+ 1 2 None))
  "

  `(try
     (do ~@body)
     (except [e Exception]
       (setv current-file-path (if (in "__file__" (globals))
                                   (do
                                     (import os)
                                     (os.path.abspath __file__))
                                   "<non-file>"))
       (print (.format "Exception occured in do-block '{}' of {}" (str '~name-symbol) current-file-path))
       (raise e))))


(comment
  ;; it's not used
  (defmacro import* [&rest args]
    (import [hy [HyExpression HyList]])
    (import [.hy-util [hylist hylist? hyexpr? hystring?]])

    (defn make-body [expr]
      (cond
        [(hylist? expr)
         (HyList (map make-body expr))]
        [(keyword? expr)
         expr]
        [(hystring? expr)
         (HySymbol expr)]
        [(hyexpr? expr)
         (HySymbol (eval expr))]
        [True expr]))

    (setv body (HyExpression (map make-body args)))
    `(eval '~(+ '(import) body))))


(defmacro import-by-str [&rest args]
  (import [hy [HyExpression HyList]])
  (import [.hy-util [hyexpr hylist hylist? hystring? hyexpr?]])

  (defn make-body [expr]
    (cond
      [(hylist? expr)
       (HyList (map make-body expr))]
      [(keyword? expr)
       expr]
      [(or (symbol? expr) (hystring? expr) (hyexpr? expr))
       (hyexpr 'unquote `(HySymbol ~expr))]
      [True expr]))

  (setv body (HyExpression (map make-body args)))
  (setv final-expr (hyexpr'quasiquote (+ '(import) body)))
  `(eval ~final-expr #_(globals)))


(defmacro do-call [&rest args]
  "example: (list (do-call (for [x (range 10)] (yield (* x x)))))"
  `((fn []
      ~@args
      )))


(defmacro do-let [binding &rest args]
  "example: (list (do-let [y 10] (for [x (range 10)] (yield (* x y)))))"
  (assert (even? (len binding)))
  (setv [params arguments] (map HyExpression (zip #* (partition binding))))
  `((fn [~@params]
      ~@args
      ) ~@arguments))


(defmacro eval-locally-once [expr]
  "evaluate the expression only once in the current local scope"
  (setv result (HySymbol (mangle (gensym))))
  `(do
     (unless (in (str '~result) (locals))
       (setv ~result ~expr))
     ~result))

(defmacro! test-or [o!test &rest args]
  (import [hy [HyExpression]])
  (setv value (gensym))
  (defn get-arg-expr [arg]
    `(~g!test (setx ~value ~arg)))
  
  `(do (or ~@(HyExpression (map get-arg-expr args)))
       ~value))
