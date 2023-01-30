exp=$1
qsub -cwd -S /bin/bash -l qp=cuda-low,tests=0,mem_grab=0M,osrel="*",gpuclass="volta",not_host=air212 -o LOGs/log_${exp} train.sh
