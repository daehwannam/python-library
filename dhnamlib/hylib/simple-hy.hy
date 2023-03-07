
;; This code is a simple implementation of Hy
;; inspired by Peter Norvig's article.
;;
;; https://norvig.com/lispy.html
;; https://norvig.com/lis.py

(import [hy [HyExpression HySymbol HyDict HyList HySet HyDict]])
(import hy.models)
(import [hy.contrib.hy-repr [hy-repr]])

(import [dhnamlib.hylib.common [hysymb hyexpr hylist hyset
                                       hyexpr? hylist? hysequence?]])
(import [dhnamlib.pylib [iteration :as iter-util]])

(import [dhnamlib.hylib [eval :as eval-util]])
(import [dhnamlib.hylib.linkedlist [AssociationList]])


(defmacro make-local-eval-hy []
  '(do
     (import [dhnamlib.hylib [simple-hy]])
     (fn [expr]
       (simple-hy.evaluate-hy expr (make-binding :global-namespace (globals)
                                                 :local-namespace (locals))))))

(defn eval-hy [expr &optional global-namespace local-namespace]
  (evaluate-hy expr (make-binding :global-namespace global-namespace
                                  :local-namespace local-namespace)))

(defn eval-hy [expr &optional global-namespace local-namespace]
  (setv args (list (remove none? [global-namespace local-namespace])))
  (setv binding (make-binding global-namespace local-namespace))
  (evaluate-hy expr binding))

(defn evaluate-hy [expr binding]
  (if (hysequence? expr)
      (or
        (unless (empty? expr)
          (setv [head #* dependents] expr)
          (cond
            [(= head 'fn) (eval-fn dependents binding)]
            [(= head 'let) (eval-let dependents binding)]
            [(= head 'if) (eval-if dependents binding)]
            [(= head ',) (eval-tuple dependents binding)]
            [(hyexpr? expr) (eval-call head dependents binding)]
            [(in head '[do setv])
             (raise (Exception "The components causing side effects aren't going to be implemented"))]))
        (eval-sequence expr binding))
      (or
                                ; variable
        (try
          (eval-variable expr binding)
          (except [NonExistingVariable]))
                                ; literal
        (eval-literal expr))
      )
  )

(defn eval-fn [dependents binding]
  (setv [param-symbols #* body-exprs] dependents)
  (assert (= (len body-exprs) 1) "More than 1 expression is not allowed yet")

  (fn [&rest args]
    (assert (= (len param-symbols) (len args)))
    (setv new-binding binding)
    (for [[param-symbol arg] (zip param-symbols args)]
      (setv new-binding (update-binding new-binding param-symbol arg)))

    (setv body-expr (first body-exprs))
    (evaluate-hy body-expr new-binding)))

(defn eval-let [dependents binding]
  (setv [var-value-list body-exprs] dependents)

  (assert (even? (len var-value-list)))
  (setv var-value-pairs (partition var-value-list))

  (setv new-binding binding)
  (for [[var-symbol value-expr] var-value-pairs]
    (setv new-binding (.update binding var (evaluate-hy value))))

  (assert (= (len body-exprs) 1) "More than 1 expression is not allowed yet")
  (setv body-expr (first body-expr))

  (evaluate-hy body-expr binding))

(defn eval-if [dependents]
  (setv [condition-expr true-statement-expr false-statement-expr]
        dependents)

  (if (evaluate-hy condition-expr binding)
      (evaluate-hy true-statement-expr binding)
      (evaluate-hy false-statement-expr binding)))

(defn eval-tuple [dependents binding]
  (tuple (gfor dependent dependents (evaluate-hy dependent binding))))

(defn eval-sequence [expr binding]
  (setv hy-py-cls-pairs
        [hy.models.HyList list
         hy.models.HyDict dict
         hy.models.HySet set])
  (for [[hy-cls py-cls] (partition hy-py-cls-pairs)]
    (when (instance? hy-cls expr)
      (return (py-cls (gfor elem expr (evaluate-hy elem binding))))))
  (setv py-cls-tuple (tuple (gfor [hy-cls py-cls] (partition hy-py-cls-pairs) py-cls)))
  (when (instance? py-cls-tuple expr)
    (return expr))
  (raise (Exception "Unknown sequence")))

(defclass NonExistingVariable [Exception])

(setv no-value '__no-value__)

(defn eval-variable [var-symbol binding]
  (setv var-name (get-name var-symbol))
  (cond
    [(in var-name __builtins__)
     (get __builtins__ var-name)]
    [(in var-name (setx global-namespace (.get binding '__global-namespace__)))
     (get global-namespace var-name)]
    [(in var-name (setx local-namespace (.get binding '__local-namespace__)))
     (get local-namespace var-name)]
    [(is-not (setx value (.get binding var-symbol :value no-value)) no-value)
     value]
    [True
     (raise (NonExistingVariable "Variable doesn't exist."))]))

(defn eval-literal [expr]
  (setv hy-py-cls-pairs
        (tuple (partition
                 [hy.models.HyString str
                  hy.models.HyInteger int
                  hy.models.HyFloat float])))
  (for [[hy-cls py-cls] hy-py-cls-pairs]
    (when (instance? hy-cls expr)
      (return (py-cls expr))))
  (setv py-cls-tuple (tuple (gfor [hy-cls py-cls] hy-py-cls-pairs py-cls)))
  (when (and (not (symbol? expr))
             (instance? py-cls-tuple expr))
    (return expr))
  (try
    (return (getattr hy.core.shadow (mangle (name expr))))
    (except [AttributeError]))
  (raise (Exception f"Unknown literal {expr}")))

(defn eval-call [head dependents binding]
  (setv func (evaluate-hy head binding))
  (setv kwarg-begin-idx (iter-util.index dependents hy.HyKeyword
                                        :key (fn [x] (type x))
                                        :default (len dependents)))
  (setv arg-exprs (cut dependents 0 kwarg-begin-idx))
  (setv kw-arg-expr-pairs (tuple (partition (cut dependents kwarg-begin-idx))))
  (do
    (assert (even? (- (len dependents) (len arg-exprs))))
    (for [[kw arg-expr] kw-arg-expr-pairs]
      (assert (instance? hy.HyKeyword kw))))

  (setv args (lfor arg-expr arg-exprs (evaluate-hy arg-expr binding)))
  (setv kwargs (dfor [kw arg-expr] kw-arg-expr-pairs
                     [(get-name kw) (evaluate-hy arg-expr binding)]))

  (try
    (func #* args #** kwargs)
    (except [TypeError]
      (breakpoint)
      (print)
      ))
  )

(defn get-name [unit]
  "unit is a symbol or keyword"
  (mangle (name unit)))

(defmacro with-brackets [components-l-r-brackets &rest body]
  (setv [components l-bracket r-bracket]
        components-l-r-brackets)
  `(do
     (.append ~components ~l-bracket)
     ~@body
     (.append ~components ~r-bracket)
     )
  )

(do
  ;; scope

  (defn make-binding [&optional global-namespace local-namespace]
    (unless global-namespace (setv global-namespace (globals)))
    (unless local-namespace (setv local-namespace {}))
    (AssociationList.from-pairs ['__global-namespace__ global-namespace]
                                ['__local-namespace__ local-namespace]))

  (defn update-binding [binding var-symbol value]
    (.update binding var-symbol value))
  )


(when (= __name__ "__main__")
  (print "test")
  (print (eval-hy '(tuple (map (fn [x] (abs (- (** x 2)))) (, 1 2 3 4))) (globals)))
  (for [arg (range 2000)]
    (eval-hy '(tuple (map (fn [x] (abs (- (** x 2)))) (, 1 2 3 4))) (globals) (locals))
    ;; (eval '(tuple (map (fn [x] (abs (- (** x 2)))) (, 1 2 3 4))) (globals))
    ;; (eval-util.pyeval (disassemble '(tuple (map (fn [x] (abs (- (** x 2)))) (, 1 2 3 4))) :codegen True) (globals) (locals))
    ;; (eval-util.pyeval "tuple(map(lambda x: abs(- x ** 2), (1,2,3,4)))")
    ;; (disassemble '(tuple (map (fn [x] (abs (- (** x 2)))) (, 1 2 3 4))))
    ))
