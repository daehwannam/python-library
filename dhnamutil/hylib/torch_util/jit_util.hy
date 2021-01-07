
(eval-and-compile
  (import [dhnamutil.dhnamutil.hylib.hyutil [replace-symbol-recursively]]))


(defmacro! inline-map [o!func seq]
  `(do
     (setv ~g!output [])
     (for [~g!elem ~seq]
       (.append ~g!output (~g!func ~g!elem)))
     ~g!output))


(defmacro! inline-filter-or-remove [func seq when-or-unless]
  (if (= (first func) 'fn)
      (do
        (setv [fn-symbol [fn-param] fn-body] func)
        (setv new-fn-param (gensym))
        (setv new-fn-body (replace-symbol-recursively fn-body fn-param new-fn-param))
        `(do
           (setv ~g!output [])
           (for [~new-fn-param ~seq]
             (~when-or-unless ~new-fn-body
               (.append ~g!output ~new-fn-param)))
           ~g!output))
      `(do
         (setv ~g!output [])
         (setv ~g!func ~func)
         (for [~g!elem ~seq]
           (~when-or-unless (~g!func ~g!elem)
             (.append ~g!output ~g!elem)))
         ~g!output)))

(defmacro! inline-filter [func seq]
  `(inline-filter-or-remove ~func ~seq when))


(defmacro! inline-remove [func seq]
  `(inline-filter-or-remove ~func ~seq unless))


;; (defmacro! inline-lfor [elem o!seq &rest args]
;;   (import [hy [HyExpression]])
;;   (setv elem-expr (last args))
;;   (setv optional-args (HyExpression (butlast args)))
;;   `(do
;;      (setv ~g!output [])
;;      (for [~elem ~g!seq]
;;        )
;;      )
;;   )
