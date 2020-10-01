# Triplet

#### To install with development libraries

```
$ pip install virtualenv # install the virtual env
$ virtualenv myvirtualenvname # create the virtual env
$ source myvirtualenvname/bin/activate # activate the environment
```
Once your virtualenv is activated by the source myvirtualenvname/bin/activate you can install
the package by the following command.

Dev option will also install pytest so you can add and run your tests.

if you are using zsh shell you have to use escacpe characters for brackets.
```
$ pip install -e .\[dev\]
```
otherwise you can install it as follows
```
$ pip install -e .[dev]
```

#### To install without development libraries
While you are an the root directory just run the following command.
This will exclude pytest package from the installation.

```
$ pip install -e .
```
