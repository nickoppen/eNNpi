EXE="/home/pi/git/piGit/eNNpi/Release"
#EXE="$HOME/git/piGit/eNNpi/Debug"
for i in `ls *.enn`; do 
	$EXE/eNNpi -q+ -n $i -test ./shuttle7_5_Sets1-4.tr > $i.res
done
