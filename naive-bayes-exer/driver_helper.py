import getopt
import sys


def main(argv):
    alpha = 0
    reduce = True
    word_freq = 0
    try:
        opts, _ = getopt.getopt(
            argv, 'ha:r:v:', ["alpha=", "reduce=", "vocab_count_class="])
    except getopt.GetoptError:
        print('Invalid argument')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('driver.py -a <alpha> -r <reduce(T|F)> -v <word frequency>')
            sys.exit(2)
        elif opt in ('-a', '--alpha'):
            alpha = arg
        elif opt in ('-r', '--reduce'):
            if arg.lower() in ('t', 'true'):
                reduce = True
            elif arg.lower() in ('f', 'false'):
                reduce = False
        elif opt in ('-v', '--vocab_count_class'):
            word_freq = arg

    return alpha, reduce, word_freq
