
(import [hy.lex [hy-parse]])
(import [hy.compiler [hy-compile]])
(import [hy.contrib.hy-repr [hy-repr]])

(import astor)

(import [..pylib [filesys]])


(setv default-module "__main__")


(defn hytext-to-pytext [hytext &optional [module default-module]]
  (astor.to_source (hy-compile (hy-parse hytext) module)))


(defn hyfile-to-pyfile [hyfile-path pyfile-path &optional [module default-module]]
  (filesys.write_file pyfile-path (hytext-to-pytext (filesys.read_file hyfile-path) module)))
