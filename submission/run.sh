#!/bin/bash
# m640_a0p1: card run through the official harness. Usage: ./run.sh [practice|coco1k] [repeats] [extra evaluate.py args]
# Output: results/<set>_<UTC time>.json + .log. Card access only via tt-lock; check `tt-who` first.
set -e
HERE=$(cd "$(dirname "$0")" && pwd); SET=${1:-practice}; REP=${2:-3}; shift $(( $# > 2 ? 2 : $# )) || true
if [ "$SET" = practice ]; then IM=~/challenge/practice/images; AN=~/challenge/practice/instances.json
else IM=~/work/data/coco1k/images; AN=~/work/data/coco1k/instances.json; fi
cd ~/tt-metal && source python_env/bin/activate
export TT_METAL_HOME=~/tt-metal PYTHONPATH=~/tt-metal:/home/user/challenge/harness
OUT=$HERE/results/${SET}_$(date -u +%m%d_%H%M%S)
tt-lock ${TT_USER:-chanyoung} python ~/challenge/harness/evaluate.py --submission $HERE/submission.py \
  --images $IM --annotations $AN --repeats $REP --team hawkeyett --out $OUT.json "$@" > $OUT.log 2>&1
python3 -c "import json;r=json.load(open('$OUT.json'));t=r['timing'];print('m640_a0p1', '$SET', 'ms', t['ms_per_frame'], 'each', t.get('ms_per_frame_each_run'), 'disturbed', t.get('disturbed'), 'mAP50', r['accuracy']['mAP_50'])"
