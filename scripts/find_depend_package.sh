#!/bin/bash

logfile=/home/ubuntu/Desktop/log
ret=""
function getDepends()
{
   echo "fileName is" $1>>$logfile
   # use tr to del < >
   ret=`apt-cache depends $1|grep Depends |cut -d: -f2 |tr -d "<>"`
   echo $ret
}
# 需要获取其所依赖包的包
libs="libatlas-base-dev"                  # 或者用$1，从命令行输入库名字

# download libs dependen. deep in 3
i=0
while [ $i -lt 3 ] ;
do
    let i++
    # download libs
    newlist=" "
    for j in $libs
    do
        added="$(getDepends $j)"
        newlist="$newlist $added"
		echo $added
        #apt-get install -d -y $added
    done

    libs=$newlist
done
