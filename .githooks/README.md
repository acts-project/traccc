# traccc git hooks

This directory stores some useful hooks for traccc developers. To install one of
these hooks, run the following command (from the repository root):

    ln -s $(pwd)/.githooks/pre-commit .git/hooks

If you are using a modern version of coreutils and you need relative paths, for
example if your mount points may change, use:

    ln -rs .githooks/pre-commit .git/hooks

To install all hooks in this repository, you can also run:

    git config --local core.hooksPath .githooks

## Requirements

These hooks require clang-format to be installed, and findable. Either
`clang-format` needs to be a binary findable by bash, or the
`CLANG_FORMAT_BINARY` environment variable must be set to a clang-format binary.
