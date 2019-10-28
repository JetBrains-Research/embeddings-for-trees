from argparse import ArgumentParser, Namespace


def main(args: Namespace) -> None:
    labels = []

if __name__ == '__main__':
    argument_parser = ArgumentParser(description='collect labels based on batched data')
    argument_parser.add_argument('--path', type=str, required=True, help='path to folder with batches')
    argument_parser.add_argument('--output', type=str, required=True, help='path to output pickle')

    main(argument_parser.parse_args())