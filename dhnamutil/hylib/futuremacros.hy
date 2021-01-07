
(defmacro setx [var value]
  `(do (setv ~var ~value) ~var))
