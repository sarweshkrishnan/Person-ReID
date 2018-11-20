mkdir mixed

for dir in $scratch/duke/pytorch/*/ ; do
  cd $dir
  echo $dir
  for id in */ ; do
    cd $id
    for img in * ; do
      mv "$img" "1$img"
    done
    cd ..
    mv "$id" "1$id"
  done
done
