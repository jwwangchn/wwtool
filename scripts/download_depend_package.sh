package=libboost-all-dev

apt-get install --reinstall -d `apt-cache depends $package | grep Depends |cut -d: -f2 |tr -d "<>"` $package
