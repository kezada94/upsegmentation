import argparse


class FloatAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) not in [1, 3]:
            raise argparse.ArgumentTypeError("Must provide either one float or three floats.")
        setattr(namespace, self.dest, values)
