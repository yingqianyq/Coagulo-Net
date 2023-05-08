#!/bin/bash
for ((i=1; i<=10; i=i+1))
do 
   python ode.py --noisescale 0.1 --output output/res_1.txt
   python ode.py --noisescale 0.01 --output output/res_2.txt
   python ode.py --noisescale 0.001 --output output/res_3.txt
   python ode.py --noisescale 0.0001 --output output/res_4.txt
   python ode.py --noisescale 0.00001 --output output/res_5.txt
   python ode.py --noisescale 0.000001 --output output/res_6.txt
done
