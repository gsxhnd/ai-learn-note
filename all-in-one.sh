##/bin/bash

dist="one/all.md"
echo $dist

cat one/start.md > $dist
cat README.md >> $dist
cat 00/README.md >> $dist
cat 00/01.md >> $dist
cat 00/02.md >> $dist

cat lstm/README.md >> $dist
cat lstm/01.md >> $dist
cat lstm/02.md >> $dist
cat lstm/03.md >> $dist