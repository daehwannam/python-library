
(import [..pylib [filesys]])
(import [..pylib.decoration [*]])

(defn file-cache [file-path-arg-name &kwonly load-func save-func]
  (import os)

  (defn file-cache-decorator [func]
    #@((functools.wraps func)
        (defn file-cache-func [&rest args &kwargs kwargs]
          (setv file-path (get kwargs (mangle file-path-arg-name)))
          (del (get kwargs (mangle file-path-arg-name)))

          (if (and file-path (os.path.isfile file-path))
              (setv obj (load-func file-path))
              (do
                (setv obj (func #* args #** kwargs))
                (filesys.mkpdirs_unless_exist file-path)
                (save-func obj file-path)))
          obj))

    file-cache-func)

  file-cache-decorator)
