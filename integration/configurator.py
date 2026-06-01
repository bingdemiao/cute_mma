"""Poor-man's configurator. Borrowed from Karpathy's nanoGPT verbatim.

Usage::

    $ python train.py config/train_shakespeare_char.py --batch_size=32

Each positional argument that ends with ``.py`` is exec'd in the current
namespace, overriding any defaults defined above the ``exec(open(...).read())``
call site. ``--key=value`` flags are also exec'd in the same namespace,
type-coerced via ``literal_eval``.

The point is to keep the training script a single global namespace whose
defaults can be overridden without an argparse boilerplate or a YAML file.
"""

import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    if "=" not in arg:
        # Treat as a config file path
        assert not arg.startswith("--")
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        with open(config_file) as f:
            exec(f.read())
    else:
        # KEY=VALUE flag
        assert arg.startswith("--")
        key, val = arg.split("=", 1)
        key = key[2:]
        if key in globals():
            try:
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                attempt = val
            assert type(attempt) == type(globals()[key]), (
                f"--{key}={val!r} (type {type(attempt).__name__}) does not match "
                f"existing type {type(globals()[key]).__name__}"
            )
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
