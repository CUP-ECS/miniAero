scale="$1"
xy=`echo "2.0 * $scale" | bc -l`
z=`echo "1.0 * $scale" | bc -l`
xypts=`echo "256 * $scale" | bc -l`
zpts=`echo "16 * $scale" | bc -l`
echo 2
echo $xy $xy $z 30.0
echo $xypts $xypts $zpts
echo 1600
echo 2e-6 
echo 1
echo 100
echo 1
echo 0
