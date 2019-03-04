import getopt


def main(argv):
    alpha = 0
    word_freq = 0
    try:
        opts, _ = getopt.getopt(argv, 'ha:w:', ["alpha=", "word_freq="])
    except getopt.GetoptError:
        print('Invalid argument')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('driver.py -a <alpha> -wf <word frequency>')
        elif opt in ('-a', '--alpha'):
            alpha = arg
        elif opt in ('-w', '--word_freq'):
            word_freq = arg

    return alpha, word_freq
