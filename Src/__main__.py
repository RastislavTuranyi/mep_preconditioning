import argparse
import logging

from Src.main import precondition_path_ends, read_input


class InputError(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description='Precondition start and end images prior to MEP interpolation.')

    parser.add_argument('-i', '--input', action='append', nargs='*', type=str)
    parser.add_argument('-s', '--start', action='extend', nargs='*', type=str)
    parser.add_argument('-e', '--end', action='extend', nargs='*', type=str)
    parser.add_argument('-o', '--output', action='extend', nargs='*', type=str, default=['preconditioned.xyz'])
    parser.add_argument('--stepwise-output', action='store_true')

    args = parser.parse_args()

    if args.input is not None:
        start, end, both = None, None, None

        if len(args.input) == 1:
            input = args.input[0]
            if isinstance(input, str):
                # Input is one file containing both images
                both = input
            else:
                if len(input) == 1:
                    # Input is one file containing both images
                    both = input[0]
                elif len(input) == 2:
                    # Input is two files, each containing one of the images
                    start = input[0]
                    end = input[1]
                else:
                    raise InputError()

        # There are either two strings, two lists, or a string and a list as an input
        elif len(args.input) == 2:
            start = args.input[0]
            end = args.input[1]

            if isinstance(start, str):
                start = [start]
            if isinstance(end, str):
                end = [end]

        # Option -i has been called three or more times
        else:
            raise InputError()

    elif args.start is not None and args.end is not None:
        start = args.start
        end = args.end
        both = None
    else:
        raise Exception()

    reactant, product, _, _, reactant_indices, product_indices = read_input(start, end, both)
    precondition_path_ends(reactant, product, reactant_indices, product_indices)


if __name__ == '__main__':
    main()