
(import [..pylib.package-util [*]])
(import [.hy-util [weak-mangle]])
(import importlib)


(defn import-module [module-name &optional [package None]]
  ;; weak-mangle is used instead of 'mangle which collapse dot(.) sign
  (if (none? package)
      (importlib.import-module (weak-mangle module-name))
      (importlib.import-module (weak-mangle module-name) (weak-mangle package))))

(comment
  ;; moved to package-util.py

  (defn get-ancestor [full-name gap]
    "'full-name' is the full name of module or package"

    (assert (pos? gap))
    (.join "." (cut (.split full-name ".") 0 (- gap))))


  (defn get-parent [package]  ; parent-package
    "a.b.c --> a.b"

    (get-ancestor package 1))


  (defn join [&rest args]  ; join-names
    (.join "." args))
  )

(defn import-from-module [module-name obj-names]
  (if (instance? str obj-names)
      (do
        (setv obj-names [obj-names])
        (setv sole-obj True))
      (setv sole-obj False))

  (setv module (import-module module-name))
  (setv objs (tuple (gfor obj-name obj-names
                          (getattr module (mangle obj-name)))))

  (if sole-obj (first objs) objs))
