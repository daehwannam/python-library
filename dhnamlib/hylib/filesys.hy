
(import hy)
(import [..pylib [filesys :as pyfilesys]])
(import [..pylib.filesys [*]])


(defn python-load [path]
  (pyfilesys.python-load path (vars hy)))


(defn hy-read [path]
  (read-str (pyfilesys.read-text path)))

(defn hy-load [path]
  (eval (read-str (pyfilesys.read-text path))))


(defmacro get-current-file-path []
  '(do
     (import os)
     (os.path.abspath __file__)))
