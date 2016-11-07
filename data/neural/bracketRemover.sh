 #!/bin/bash   


gsed -s 's/[\[ \ ]//g' "./letters_exported.txt" > "./.temp2.txt"
gsed -s 's/\]//g' "./.temp2.txt" > "./letters_not_shuffled.txt"

