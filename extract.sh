#!/bin/bash

grep 'Electrostatics sSAPT0' out.txt > new.txt
grep 'Exchange sSAPT0' out.txt > new.txt
grep 'Induction sSAPT0' out.txt > new.txt
grep 'Dispersion sSAPT0' out.txt > new.txt
grep 'Total sSAPT0' out.txt > new.txt


