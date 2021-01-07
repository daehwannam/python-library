
(import [inspect [isclass]])
(import [hy.contrib.hy-repr [hy-repr hy-repr-register]])

(import [..pylib [linked_list :as pyll]])
(import [..pylib.linked_list [LinkedList]])
(import [..pylib.linked_list [*]])


(defn linked-list-repr [ll]
  (hy-repr (list ll)))


(hy-repr-register LinkedList linked-list-repr)

(for [obj-name (dir pyll)]
  (setv obj (getattr pyll obj-name))
  (when (and (isclass obj) (issubclass obj LinkedList))
    (hy-repr-register obj linked-list-repr)))
