#!/bin/bash
# utility script for uploading generated docs to GitHub pages
set -e
git checkout master
doxygen Doxyfile
git checkout -b gh-pages-helper
git add -f docs/html
git commit -m 'Added doxygen docs'
git push github :gh-pages
git subtree push --prefix docs/html github gh-pages
git checkout master
git branch -D gh-pages-helper
