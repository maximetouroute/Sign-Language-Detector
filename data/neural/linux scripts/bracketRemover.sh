 #!/bin/bash   


sed -s 's/[\[ \ ]//g' "./letters_exported.txt" > "./.temp2.txt"
sed -s 's/\]//g' "./.temp2.txt" > "./letters_not_shuffled.txt"

