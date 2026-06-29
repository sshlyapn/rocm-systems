cd /home/sshliapn/code/llama.cpp_pm4_epxeriments/vk_gap_test
export HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/home/sshliapn/code/rocm-libraries/projects/clr/build-gap/hipamd/lib
unset HIP_PM4_GRAPH
gfx(){ amd-smi metric -g 0 2>/dev/null | awk "/CLOCK:/{c=1} c&&/CLK:/{print \$2;exit}"; }
busy(){ out=/tmp/sw_$1_$2; rm -rf $out
  rocprofv3 --kernel-trace --output-format csv -d $out -o t -- ./hip_gap_graph.x $1 16384 $2 >/dev/null 2>&1
  f=$(find $out -name "*kernel_trace.csv"|head -1)
  python3 -c "import csv,sys,statistics;r=list(csv.DictReader(open(sys.argv[1])));d=sorted(float(x[\"End_Timestamp\"])-float(x[\"Start_Timestamp\"]) for x in r);print(round(statistics.median(d)/1000,2))" "$f"; }
e2e(){ ./hip_gap_graph.x $1 16384 $2 | sed -n "s/.*e2e_us=\([0-9.]*\).*/\1/p"; }
echo "GFX_MHz $(gfx)"
echo "spin K busy AQL PM4"
for sk in "250 1500" "500 1500" "750 1000" "1000 800" "1500 800" "2000 600" "3000 400" "5000 300" "10000 150" "25000 80" "50000 50" "100000 40" "200000 30" "210000 30"; do
  set -- $sk; s=$1; k=$2
  echo "$s $k $(busy $s $k) $(e2e $s $k) $(HIP_PM4_GRAPH=1 e2e $s $k)"
done
