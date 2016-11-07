#!/bin/bash   
gsed -s 's/[\[ \ ]//g' "./data/neural/letters_exported.txt" > "./data/neural/.temp2.txt"
gsed -s 's/\]//g' "./data/neural/.temp2.txt" > "./data/neural/letters_not_shuffled.txt"