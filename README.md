# Minigia: Synergia mini apps

* Synergia
	* [https://cdcvs.fnal.gov/redmine/projects/synergia2](https://cdcvs.fnal.gov/redmine/projects/synergia2)

* vectorclass: This distribution includes the vector class library.
	* [http://www.agner.org/optimize/#vectorclass](http://www.agner.org/optimize/#vectorclass)
	* see vectorclass/license.txt

## Building
    mkdir build 
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release -DDEFINES='-DGSV_AVX' \
        -DCMAKE_CXX_COMPILER=/usr/local/bin/g++-6 ..

    make
