
;; This code is a simple implementation of Hy
;; inspired by Peter Norvig's article.
;;
;; https://norvig.com/lispy.html
;; https://norvig.com/lis.py

(import [hy [HyExpression HySymbol HyDict HyList HySet HyDict]])
(import hy.models)
(import [hy.contrib.hy-repr [hy-repr]])
(import hy.core.shadow)

(import [dhnamutil.dhnamutil.hylib.hyutil [hysymb hyexpr hylist hyset
                                       hyexpr? hylist? hysequence?]])
(import [dhnamutil.dhnamutil.pylib [iterutil]])

(import [dhnamutil.dhnamutil.hylib [evalutil]])
(import [dhnamutil.dhnamutil.hylib.linkedlist [AssociationList]])


(defmacro make-local-eval-hy []
  '(do
     (import [dhnamutil.dhnamutil.hylib [fast-eval]])
     (fn [expr]
       (fast-eval.eval-hy expr (globals) (locals)))))

(defmacro with-brackets [components-l-r-brackets &rest body]
  (setv [components l-bracket r-bracket]
        components-l-r-brackets)
  `(do
     (.append ~components ~l-bracket)
     ~@body
     (.append ~components ~r-bracket)
     )
  )

(defn eval-hy [expr &optional global-namespace local-namespace]
  (setv args (list (remove none? [global-namespace local-namespace])))
  (setv py-code (hy2py expr))
  (evalutil.pyeval py-code #* args))

(defn hy2py [expr]
  (setv components [])
  (accumulate-py expr components)
  (.join "" components))

(defn accumulate-py [expr components]
  (if (hysequence? expr)
      (or
        (unless (empty? expr)
          (setv [head #* dependents] expr)
          (cond
            [(= head 'fn) (accumulate-fn dependents components)]
            [(= head 'let) (accumulate-let dependents components)]
            [(= head 'if) (accumulate-if dependents components)]
            [(= head ',) (accumulate-tuple dependents components)]
            [(hyexpr? expr) (accumulate-call head dependents components)]
            [(in head '[do setv])
             (raise (Exception "The components causing side effects aren't going to be implemented"))]))
        (accumulate-sequence expr components))
      (or
        (cond
          [(symbol? expr)
           (accumulate-variable expr components)]
          [True
           (accumulate-literal expr components)])
        )
      )
  )

(defn accumulate-fn [dependents components]
  (setv [params #* body-exprs] dependents)

  (assert (= (len body-exprs) 1) "More than 1 expression is not allowed yet")
  (setv body-expr (first body-exprs))

  (with-brackets
    [components "(" ")"]
    (.append components (.format "lambda {}:" (.join ", " (map get-name params))))
    (accumulate-py body-expr components)))

(defn accumulate-let [dependents components]
  "it's same with common lisp's let"

  (setv [var-value-list #* body-exprs] dependents)

  (assert (= (len body-exprs) 1) "More than 1 expression is not allowed yet")
  (setv body-expr (first body-exprs))

  (assert (even? (len var-value-list)))
  (setv var-value-pairs (tuple (partition var-value-list)))
  (setv [var-exprs value-exprs] (zip #* var-value-pairs))

  ;; (breakpoint)
  (accumulate-py `(fn [~@var-exprs] ~body-expr) components)
  (with-brackets
    [components "(" ")"]
    (for [value-expr value-exprs]
      (accumulate-py value-expr components)
      (.append components ","))))

(defn accumulate-if [dependents components]
  (setv [condition-expr true-statement-expr false-statement-expr]
        dependents)

  (with-brackets
    [components "(" ")"]
    (accumulate-py true-statement-expr components)
    (.append components " if ")
    (accumulate-py condition-expr components)
    (.append components " else ")
    (accumulate-py false-statement-expr components)))

(defn accumulate-tuple [dependents components]
  (with-brackets
    [components "(" ")"]

    (for [dependent dependents]
      (accumulate-py dependent components)
      (.append components ","))))

(defn accumulate-sequence [expr components]
  (setv hy-py-cls-pairs
        [hy.models.HyList list
         hy.models.HyDict dict
         hy.models.HySet set])
  (for [[hy-cls py-cls] (partition hy-py-cls-pairs)]
    (when (or (instance? hy-cls expr) (instance? py-cls expr))
      (.append components py-cls.__name__)
      (with-brackets
        [components "(" ")"]
        (accumulate-tuple expr components))
      (return))))

(defn accumulate-variable [var-symbol components]
  (.append components (get-name var-symbol)))

(defn accumulate-literal [expr components]
  (setv value
        (cond [(instance? hy.models.HyString expr) (string expr)]
              [(instance? hy.models.HyInteger expr) (int expr)]
              [(instance? hy.models.HyFloat expr) (float expr)]
              ;; [(hasattr hy.core.shadow (mangle expr)) f"hy.core.shadow.{(mangle expr)}"]
              [True expr]))
  (.append components (repr value)))

(defn accumulate-call [head dependents components]
  (setv kwarg-begin-idx (iterutil.index dependents hy.HyKeyword
                                        :key (fn [x] (type x))
                                        :default (len dependents)))
  (setv arg-exprs (cut dependents 0 kwarg-begin-idx))
  (setv kw-arg-expr-pairs (tuple (partition (cut dependents kwarg-begin-idx))))

  (when (.startswith head '.)
    (setv first-arg (first arg-exprs))
    (setv arg-exprs (tuple (rest arg-exprs)))
    (accumulate-py first-arg components))

  (.append components (get-name head))
  (with-brackets
    [components "(" ")"]
    (for [arg-expr arg-exprs]
      (accumulate-py arg-expr components)
      (.append components ","))
    (for [[kw arg-expr] kw-arg-expr-pairs]
      (.append components (.format "{}=" (get-name kw)))
      (accumulate-py arg-expr components)
      (.append components ","))))

(defn get-name [unit]
  "unit is a symbol or keyword"
  (defn relaxed-mangle [s] (if s (mangle s) s))
  (setv unit-name (name unit))
  (cond
    [(= unit ".") (raise (Exception "the sole dot is not supported yet"))]
    [(in "." unit-name) (.join "." (map relaxed-mangle (.split unit-name ".")))]
    [True (mangle unit-name)]))

(when (= __name__ "__main__")
  (defn g [x] (/ x 10))
  (print "test")
  [+ - ** <]  ;; functions in "hy.core.shadow" should be imported
  (print (eval-hy '(if (< 10 (+ 5 5)) 100 200)))
  (print (eval-hy '[(tuple (let [y 3] (let [ z (- (** y 2))] (map (fn [x] (g (- (+ x y z)))) (, 1 2 3 4))))) 20 30 40 50] (globals)))
  (print (eval-hy '(tuple (let [y 3] (map (fn [x] (abs (- (** x y)))) (, 1 2 3 4)))) (globals)))
  (for [arg (range 2000)]
    (import [hy.lex [hy-parse]])
    (eval-hy '(tuple (map (fn [x] (abs (- (** x 2)))) (, 1 2 3 4))) (globals))
    ;; (eval '(tuple (map (fn [x] (abs (- (** x 2)))) (, 1 2 3 4))) (globals))
    ;; (evalutil.pyeval (disassemble '(tuple (map (fn [x] (abs (- (** x 2)))) (, 1 2 3 4))) :codegen True) (globals) (locals))
    ;; (evalutil.pyeval "tuple(map(lambda x: abs(- x ** 3), (1,2,3,4)))")
    ;; (disassemble '(tuple (map (fn [x] (abs (- (** x 2)))) (, 1 2 3 4))))
    ))
