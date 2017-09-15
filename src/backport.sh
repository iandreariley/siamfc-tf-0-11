#!/bin/bash

grep -R "tensorflow" . | cut -d: -f1 | while read filename
do
  echo "file: $filename"
  cat map.csv | while IFS=, read old_val new_val
  do
    sed -i "s/$old_val/$new_val/g" "$filename"
  done
done
