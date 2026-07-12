# How to contribute

The two branches that may be used by everyone are `main` and `dev`
 - `main`: Stable branch. All tests are always supposed to pass on this branch
 - `dev`: Active development branch. This one contains the most recent contributions, but may not be working in every case.

1. Create a new branch, starting from `main` or from `dev`, with a name that fits your project.
2. During your work, commit frequently in your branch.
3. Write tests to verify that your code works. Please include these tests in the commits (we have a directory just for that!)
4. Open a merge request from your branch to a the one from which you started.

## Configuration options

The available configuration options are described in a single place: the schema file
`config/config-format.json`. To keep `config.<option>` autocompletion and static type checking
working, `wx_factory/common/configuration.py` carries a block of generated type annotations.

If you add, remove, or change the type of an option in the schema, regenerate that block:

```
python -m wx_factory.common.config_hints --write
```

and commit the change. A unit test (and the `checks` CI workflow) runs
`python -m wx_factory.common.config_hints --check` and will fail if the annotations are stale.
