#!/bin/bash
for ((i=1; i<=10; i=i+1))
do 
   python ode.py --datainput data_2k_points.mat --output output/res_2k.txt
   python ode.py --datainput data_1500_points.mat --output output/res_1500.txt
   python ode.py --datainput data_1k_points.mat --output output/res_1k.txt
   python ode.py --datainput data_500_points.mat --output output/res_500.txt
done
