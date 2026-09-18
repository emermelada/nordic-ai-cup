#!/bin/bash
# PARALLEL SEED-BLOCKS. The paired harness is single-process, so a --cpus=4 container used 1 core and
# 3 sat idle. Launch independent blocks on the free cores; each block is paired (candidate vs the
# live baseline in the SAME process/same seeds), so blocks can be pooled safely.
KEY=/Users/zaitzev/.ssh/vps_hermes
HEL=root@212.147.239.222
DEST=/opt/nac-phase-test
SSHOPT="-i $KEY -o BatchMode=yes -o ConnectTimeout=15"

# BLOCK A: H1 at higher power on 20 SEEDS NEVER USED BEFORE -> pooled with its earlier 20-seed result
# gives 40 seeds, which is what resolves whether its +314 ticks is a real 4.5% or noise.
ssh $SSHOPT $HEL "docker rm -f nac-h1b >/dev/null 2>&1; cd $DEST && docker run -d --name nac-h1b --cpus=1 -v $DEST:/work -w /work --entrypoint nice nac-eval -n 19 python experiments/w_hybrid.py 'H1_winEARN_liveSPEND:/work/experiments/wH1_winEARN_liveSPEND.json' 1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,1036,1037,1038,1039 12000 /work/deployed_params.json && echo H1B_LAUNCHED"

# BLOCK B: M_walk_250 on a second, disjoint seed block -> 30 seeds total for the memory variant
ssh $SSHOPT $HEL "docker rm -f nac-mwalkb >/dev/null 2>&1; cd $DEST && docker run -d --name nac-mwalkb --cpus=1 -v $DEST:/work -w /work --entrypoint nice nac-eval -n 19 python experiments/w_hybrid.py 'M_walk_250:/work/experiments/M_walk_250.json' 1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029 12000 /work/deployed_params.json && echo MWALKB_LAUNCHED"

sleep 6
ssh $SSHOPT $HEL "docker ps --format '{{.Names}} {{.Status}}'; docker stats --no-stream --format '{{.Name}} {{.CPUPerc}}' | head -6"