from compileutil import eval_lissp


if __name__ == '__main__':
    def delete_space(text):
        return ''.join(text.split())

    def add(*args):
        accum = args[0]
        for arg in args[1:]:
            accum += arg
        return accum

    def main():
        func = eval_lissp('(lambda (x) (add (delete_space x) (delete_space "a    b c") (delete_space "a b    c d   e")))')
        print(func('x      y z'))

    main()
