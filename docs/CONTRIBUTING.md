# Contributing

**Contributions are always welcome!**
See also the current [list of contributors](CONTRIBUTORS.md).

1. Fork this repository
2. Check out the develop branch: `git checkout develop`
3. Create your feature branch: `git checkout -b feature-something`
4. Commit your changes: `git commit -am 'Add some feature'`
5. Push to the branch: `git push origin feature-something`
6. Submit a pull request

Please keep your pull requests *feature-specific* and limit yourself
to one feature per feature branch.
Remember to pull updates from this repository before opening a new
feature branch.

If you are unsure where to add you feature into the code, please
do not hesitate to contact us.

There is no strict coding guideline, but please try to match your
code style to the code you edited or to the style in the respective
module.


## Branches

We aim to adhere to the "git flow" branching model: http://nvie.com/posts/a-successful-git-branching-model/

> Release versions (`master` branch) are tagged `major.minor.patch`, starting at `1.0.0`

Download the latest stable version from https://github.com/spirit-code/spirit/releases

The develop branch contains the latest updates, but is generally less consistently tested than the releases.


## Testing

When developing for spirit you should make use of the available tests and add tests of your own for any feature that you implement.
Tests are built and configured through `cmake` and require the `SPIRIT_BUILD_TESTS=ON` option as well as an accessible `numpy` installation.
To run all tests use:

```sh
ctest --test-dir build --output-on-failure
```


## Documentation

The documentation is built using [sphinx](https://www.sphinx-doc.org) which is configured through the `conf.py` file.
The API uses `doxygen` and `apidoc` to parse documentation from the source code, while dedicated documentation is stored in `docs` and `core/docs`.
Build requirements are listed in `docs/environment.yml` and to build the documentation run:

```sh
# building the documentation
sphinx-build -b html . _build
# looking at the documentation
python -m "http.server" -d _build
```

Alternatively you use the [`sphinx-autobuild`](https://github.com/sphinx-doc/sphinx-autobuild) project.
