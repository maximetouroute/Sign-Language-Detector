#!/bin/bash
sed -s 's/[\[ \ ]//g' "./data/neural/letters_exported.txt" > "./data/neural/.temp2.txt"
sed -s 's/\]//g' "./data/neural/.temp2.txt" > "./data/neural/letters_not_shuffled.txt"