#!/bin/bash
start=$1
number=$2
random=$3
field=$4    # cs.CV+AND+ti:detection

python arxiv.py --start ${start} --number ${number} --random ${random} --field ${field}
